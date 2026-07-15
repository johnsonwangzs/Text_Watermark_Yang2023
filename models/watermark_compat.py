"""Modern, offline-compatible implementation of Yang et al. (2023).

This module keeps the paper's watermarking pipeline while using models that are
already available in the local ``dlm`` environment:

* BERT-cased produces context-aware substitution candidates and contextual word
  similarities.
* The paper GloVe vectors provide global word similarities.
* RoBERTa-large-MNLI supplies the official entailment-based sentence score.
* NLTK supplies the official Penn Treebank POS filter and stop-word list.

The statistical encoding and detector match the official implementation: the
least-significant bit of SHA-256(previous_token + current_token) is tested
against a Bernoulli(0.5) null distribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import math
from pathlib import Path
import string
from typing import Iterable, Sequence

import nltk.data
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np
from scipy.stats import norm
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_MLM_PATH = "/data/llm/bert-base-cased"
DEFAULT_SENTENCE_MODEL_PATH = "/data/llm/roberta-large-mnli"
DEFAULT_GLOVE_PATH = "/data/llm/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"


def binary_encoding_function(token_pair: str) -> int:
    """Return the official repository's deterministic binary encoding."""

    digest = hashlib.sha256(token_pair.encode("utf-8")).hexdigest()
    return int(digest, 16) % 2


def _normalised_edit_distance(left: str, right: str) -> float:
    """Compute Levenshtein distance / max length without extra dependencies."""

    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


class GloveVectors:
    """Dependency-free reader for Gensim or Stanford GloVe text files."""

    def __init__(
        self, path: str | Path, vocabulary: set[str] | None = None
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"GloVe vectors not found: {self.path}")
        self.vectors: dict[str, np.ndarray] = {}
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8") as handle:
            first_line = next(handle).rstrip()
            header = first_line.split()
            if len(header) == 2 and all(item.isdigit() for item in header):
                dimension = int(header[1])
            else:
                dimension = len(first_line.split()) - 1
                self._add_line(first_line, dimension, vocabulary)
            for line in handle:
                self._add_line(line.rstrip(), dimension, vocabulary)
        self.dimension = dimension
        if not self.vectors:
            raise ValueError(
                f"No usable {dimension}-dimensional vectors were loaded from {self.path}"
            )

    def _add_line(
        self, line: str, dimension: int, vocabulary: set[str] | None
    ) -> None:
        if not line:
            return
        try:
            word, values = line.split(" ", 1)
        except ValueError:
            return
        if vocabulary is not None and word not in vocabulary:
            return
        vector = np.fromstring(values, sep=" ", dtype=np.float32)
        if vector.size != dimension:
            return
        length = float(np.linalg.norm(vector))
        if length > 0:
            self.vectors[word] = vector / length

    def similarity(self, left: str, right: str) -> float:
        left_vector = self.vectors.get(left.lower())
        right_vector = self.vectors.get(right.lower())
        if left_vector is None or right_vector is None:
            return 0.0
        return float(np.dot(left_vector, right_vector))

    def __len__(self) -> int:
        return len(self.vectors)


@dataclass(frozen=True)
class Substitution:
    original: str
    replacement: str
    token_index: int
    word_similarity: float
    sentence_similarity: float


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    substitutions: tuple[Substitution, ...]

    @property
    def num_substitutions(self) -> int:
        return len(self.substitutions)


@dataclass(frozen=True)
class DetectionResult:
    is_watermarked: bool
    p_value: float
    n: int
    ones: int
    z_score: float
    alpha: float
    mode: str

    def to_dict(self) -> dict[str, bool | float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class _Candidate:
    token: str
    token_id: int
    word_similarity: float
    sentence_similarity: float


class WatermarkModel:
    """English watermark embedder and detector for the current local setup.

    ``load_semantic_models=False`` is useful for fast detection, which needs
    only the tokenizer and POS tagger. Embedding and precise detection require
    the semantic models and GloVe vectors and raises a clear error if they were not loaded.
    """

    _EN_POS_WHITELIST = frozenset(
        {
            "MD", "NN", "NNS", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
            "VBZ", "RP", "RB", "RBR", "RBS", "JJ", "JJR", "JJS",
        }
    )

    def __init__(
        self,
        language: str = "English",
        mode: str = "embed",
        tau_word: float = 0.8,
        lamda: float = 0.83,
        tau_sent: float = 0.8,
        *,
        mlm_path: str | Path = DEFAULT_MLM_PATH,
        sentence_model_path: str | Path = DEFAULT_SENTENCE_MODEL_PATH,
        glove_path: str | Path = DEFAULT_GLOVE_PATH,
        top_k: int = 32,
        dropout_prob: float = 0.3,
        max_length: int = 512,
        sentence_max_length: int = 512,
        device: str | None = None,
        seed: int = 1234,
        load_semantic_models: bool = True,
    ) -> None:
        if language.lower() != "english":
            raise NotImplementedError(
                "The dlm-compatible path currently supports English only; "
                "the required Chinese WoBERT and similarity weights are not present locally."
            )
        if not 0.0 <= lamda <= 1.0:
            raise ValueError("lamda must be in [0, 1]")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= dropout_prob < 1.0:
            raise ValueError("dropout_prob must be in [0, 1)")

        self.language = "English"
        self.mode = mode
        self.tau_word = tau_word
        self.lamda = lamda
        self.tau_sent = tau_sent
        self.top_k = top_k
        self.dropout_prob = dropout_prob
        self.max_length = max_length
        self.sentence_max_length = sentence_max_length
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.mlm_path = str(mlm_path)
        self.sentence_model_path = str(sentence_model_path)
        self.glove_path = str(glove_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mlm_path, local_files_only=True, use_fast=True
        )
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for POS/wordpiece alignment")
        try:
            self.stop_words = set(stopwords.words("english"))
            pos_tag(["resource-check"])
            self._sentence_tokenizer = nltk.data.load(
                "tokenizers/punkt/english.pickle"
            )
        except LookupError as error:
            raise RuntimeError(
                "Official English filtering requires the NLTK stopwords, "
                "averaged_perceptron_tagger_eng, and punkt resources."
            ) from error

        self.mlm = None
        self.sentence_tokenizer = None
        self.sentence_model = None
        self.entailment_label_id = None
        self.glove = None
        if load_semantic_models:
            self.mlm = AutoModelForMaskedLM.from_pretrained(
                self.mlm_path, local_files_only=True
            ).to(self.device)
            self.mlm.eval()
            self.sentence_tokenizer = AutoTokenizer.from_pretrained(
                self.sentence_model_path, local_files_only=True, use_fast=True
            )
            self.sentence_model = AutoModelForSequenceClassification.from_pretrained(
                self.sentence_model_path, local_files_only=True
            ).to(self.device)
            self.sentence_model.eval()
            label2id = {
                str(label).upper(): int(label_id)
                for label, label_id in self.sentence_model.config.label2id.items()
            }
            if "ENTAILMENT" not in label2id:
                raise ValueError(
                    "The sentence model must expose an ENTAILMENT label in config.label2id"
                )
            self.entailment_label_id = label2id["ENTAILMENT"]
            bert_vocabulary = {
                token.lower()
                for token in self.tokenizer.get_vocab()
                if token.isalpha()
            }
            self.glove = GloveVectors(self.glove_path, bert_vocabulary)

    def _require_semantic_models(self) -> None:
        if self.mlm is None or self.sentence_model is None or self.glove is None:
            raise RuntimeError(
                "Embedding and precise detection require semantic models; "
                "construct WatermarkModel with load_semantic_models=True."
            )

    @property
    def backend_info(self) -> dict[str, str | int]:
        return {
            "mlm": self.mlm_path,
            "sentence_similarity": self.sentence_model_path,
            "sentence_metric": "mnli-entailment-probability",
            "entailment_label_id": (
                self.entailment_label_id
                if self.entailment_label_id is not None
                else -1
            ),
            "global_word_similarity": self.glove_path,
            "global_vocabulary_size": len(self.glove) if self.glove else 0,
            "pos_filter": "nltk-penn-treebank",
        }

    def _encoded(self, text: str, *, offsets: bool = False) -> dict:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=offsets,
        )
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(
                f"Input contains {len(encoded['input_ids'])} wordpieces, exceeding "
                f"the configured BERT limit ({self.max_length}). Split the text into "
                "shorter passages before watermarking or detection."
            )
        return encoded

    def _eligible_positions(
        self, text: str, tokens: Sequence[str], offsets: Sequence[Sequence[int]]
    ) -> dict[int, str]:
        del text, offsets
        tagged_tokens = pos_tag(list(tokens))
        eligible: dict[int, str] = {}
        for index in range(2, len(tokens) - 1):
            token = tokens[index]
            tag = tagged_tokens[index][1]
            if tag not in self._EN_POS_WHITELIST:
                continue
            if token.startswith("##") or tokens[index + 1].startswith("##"):
                continue
            if token in self.stop_words or token in string.punctuation:
                continue
            if not token.isalpha():
                continue
            eligible[index] = tag
        return eligible

    def _analyse(self, text: str) -> tuple[list[int], list[str], list, dict[int, str]]:
        encoded = self._encoded(text, offsets=True)
        input_ids = list(encoded["input_ids"])
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        offsets = encoded["offset_mapping"]
        eligible = self._eligible_positions(text, tokens, offsets)
        return input_ids, tokens, offsets, eligible

    def _sentence_pairs(self, text: str) -> list[tuple[int, int, str]]:
        spans = list(self._sentence_tokenizer.span_tokenize(text))
        if not spans and text.strip():
            spans = [(0, len(text))]
        pairs = []
        for index in range(0, len(spans), 2):
            start = spans[index][0]
            end = spans[min(index + 1, len(spans) - 1)][1]
            pairs.append((start, end, text[start:end]))
        return pairs

    def _candidate_pos_allowed(self, token: str) -> bool:
        return (
            token.isalpha()
            and token not in self.stop_words
            and token not in string.punctuation
            and pos_tag([token])[0][1] in self._EN_POS_WHITELIST
        )

    def _initial_candidates(
        self, input_ids: Sequence[int], token_index: int, original: str
    ) -> list[tuple[str, int]]:
        self._require_semantic_models()
        ids = torch.tensor([input_ids], device=self.device)
        with torch.inference_mode():
            embeddings = self.mlm.get_input_embeddings()(ids).clone()
            embeddings[:, token_index, :] = F.dropout(
                embeddings[:, token_index, :],
                p=self.dropout_prob,
                training=self.dropout_prob > 0,
            )
            logits = self.mlm(inputs_embeds=embeddings).logits[0, token_index]
            candidate_ids = torch.topk(logits, self.top_k).indices.tolist()

        candidates: list[tuple[str, int]] = []
        seen: set[str] = set()
        for candidate_id in candidate_ids:
            candidate = self.tokenizer.convert_ids_to_tokens(candidate_id)
            if (
                candidate in seen
                or candidate == original
                or candidate.startswith("##")
                or not candidate.isalpha()
                or candidate in self.tokenizer.all_special_tokens
                or _normalised_edit_distance(candidate, original) < 0.5
                or not self._candidate_pos_allowed(candidate)
            ):
                continue
            seen.add(candidate)
            candidates.append((candidate, candidate_id))
        return candidates

    def _sentence_entailment_scores(
        self, premise: str, hypotheses: Sequence[str]
    ) -> torch.Tensor:
        self._require_semantic_models()
        encoded = self.sentence_tokenizer(
            [premise] * len(hypotheses),
            list(hypotheses),
            padding=True,
            truncation=True,
            max_length=self.sentence_max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.sentence_model(**encoded).logits
        probabilities = torch.softmax(logits, dim=1)
        return probabilities[:, self.entailment_label_id]

    def _valid_candidates(
        self,
        input_ids: Sequence[int],
        token_index: int,
        original: str,
    ) -> list[_Candidate]:
        initial = self._initial_candidates(input_ids, token_index, original)
        if not initial:
            return []

        candidate_ids = [candidate_id for _, candidate_id in initial]
        batch_ids = torch.tensor(
            [list(input_ids) for _ in range(len(initial) + 1)], device=self.device
        )
        batch_ids[:-1, token_index] = torch.tensor(candidate_ids, device=self.device)

        with torch.inference_mode():
            output = self.mlm(
                input_ids=batch_ids, output_hidden_states=True, return_dict=True
            )
        layers = output.hidden_states[-8:]
        context_scores = torch.stack(
            [
                F.cosine_similarity(
                    layer[:-1, token_index, :],
                    layer[-1, token_index, :].unsqueeze(0),
                    dim=1,
                )
                for layer in layers
            ]
        ).mean(dim=0)

        current_text = self.tokenizer.decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        candidate_texts = []
        for candidate_id in candidate_ids:
            replaced = list(input_ids)
            replaced[token_index] = candidate_id
            candidate_texts.append(
                self.tokenizer.decode(
                    replaced, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            )

        sentence_scores = self._sentence_entailment_scores(
            current_text, candidate_texts
        )

        candidate_tokens = [candidate for candidate, _ in initial]
        global_scores = torch.tensor(
            [
                self.glove.similarity(original, candidate)
                for candidate in candidate_tokens
            ],
            device=self.device,
            dtype=context_scores.dtype,
        )
        word_scores = self.lamda * context_scores + (1.0 - self.lamda) * global_scores

        valid = []
        for index, (candidate, candidate_id) in enumerate(initial):
            word_score = float(word_scores[index])
            sentence_score = float(sentence_scores[index])
            if word_score >= self.tau_word and sentence_score >= self.tau_sent:
                valid.append(
                    _Candidate(candidate, candidate_id, word_score, sentence_score)
                )
        return valid

    @staticmethod
    def _match_case(candidate: str, original_surface: str) -> str:
        if original_surface.isupper():
            return candidate.upper()
        if original_surface[:1].isupper():
            return candidate.capitalize()
        return candidate

    def _embed_pair(
        self, text: str, base_offset: int
    ) -> tuple[list[tuple[int, int, str]], list[Substitution]]:
        input_ids, tokens, offsets, eligible = self._analyse(text)
        replacements: list[tuple[int, int, str]] = []
        substitutions: list[Substitution] = []

        for token_index in sorted(eligible):
            tokens[token_index] = self.tokenizer.convert_ids_to_tokens(
                input_ids[token_index]
            )
            previous = self.tokenizer.convert_ids_to_tokens(input_ids[token_index - 1])
            current = tokens[token_index]
            if binary_encoding_function(previous + current) == 1:
                continue

            valid = self._valid_candidates(input_ids, token_index, current)
            bit_one = [
                candidate
                for candidate in valid
                if binary_encoding_function(previous + candidate.token) == 1
            ]
            if not bit_one:
                continue
            selected = max(bit_one, key=lambda candidate: candidate.word_similarity)
            input_ids[token_index] = selected.token_id
            tokens[token_index] = selected.token

            start, end = offsets[token_index]
            original_surface = text[start:end]
            replacement = self._match_case(selected.token, original_surface)
            replacements.append(
                (base_offset + start, base_offset + end, replacement)
            )
            substitutions.append(
                Substitution(
                    original=original_surface,
                    replacement=replacement,
                    token_index=token_index,
                    word_similarity=selected.word_similarity,
                    sentence_similarity=selected.sentence_similarity,
                )
            )
        return replacements, substitutions

    def embed_with_report(self, text: str) -> EmbeddingResult:
        """Embed the watermark and report every accepted substitution."""

        self._require_semantic_models()
        replacements: list[tuple[int, int, str]] = []
        substitutions: list[Substitution] = []
        for start, _, sentence_pair in self._sentence_pairs(text):
            pair_replacements, pair_substitutions = self._embed_pair(
                sentence_pair, start
            )
            replacements.extend(pair_replacements)
            substitutions.extend(pair_substitutions)

        watermarked = text
        for start, end, replacement in reversed(replacements):
            watermarked = watermarked[:start] + replacement + watermarked[end:]
        return EmbeddingResult(watermarked, tuple(substitutions))

    def embed(self, text: str) -> str:
        """Compatibility wrapper matching the official repository API."""

        return self.embed_with_report(text).text

    def get_encodings_fast(self, text: str) -> list[int]:
        encodings = []
        for _, _, sentence_pair in self._sentence_pairs(text):
            _, tokens, _, eligible = self._analyse(sentence_pair)
            encodings.extend(
                binary_encoding_function(tokens[index - 1] + tokens[index])
                for index in sorted(eligible)
            )
        return encodings

    def get_encodings_precise(self, text: str) -> list[int]:
        self._require_semantic_models()
        encodings = []
        for _, _, sentence_pair in self._sentence_pairs(text):
            input_ids, tokens, _, eligible = self._analyse(sentence_pair)
            for token_index in sorted(eligible):
                if self._valid_candidates(
                    input_ids, token_index, tokens[token_index]
                ):
                    encodings.append(
                        binary_encoding_function(
                            tokens[token_index - 1] + tokens[token_index]
                        )
                    )
        return encodings

    @staticmethod
    def _test(encodings: Iterable[int], alpha: float, mode: str) -> DetectionResult:
        values = list(encodings)
        n = len(values)
        ones = sum(values)
        if n == 0:
            return DetectionResult(False, 1.0, 0, 0, 0.0, alpha, mode)
        z_score = (ones - 0.5 * n) / math.sqrt(0.25 * n)
        p_value = float(norm.sf(z_score))
        threshold = float(norm.ppf(1.0 - alpha))
        return DetectionResult(
            z_score >= threshold,
            p_value,
            n,
            ones,
            z_score,
            alpha,
            mode,
        )

    def detect(self, text: str, *, alpha: float = 0.05, precise: bool = False) -> DetectionResult:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        mode = "precise" if precise else "fast"
        encodings = (
            self.get_encodings_precise(text) if precise else self.get_encodings_fast(text)
        )
        return self._test(encodings, alpha, mode)

    def watermark_detector_fast(self, text: str, alpha: float = 0.05) -> tuple:
        result = self.detect(text, alpha=alpha, precise=False)
        return (
            result.is_watermarked,
            result.p_value,
            result.n,
            result.ones,
            result.z_score,
        )

    def watermark_detector_precise(self, text: str, alpha: float = 0.05) -> tuple:
        result = self.detect(text, alpha=alpha, precise=True)
        return (
            result.is_watermarked,
            result.p_value,
            result.n,
            result.ones,
            result.z_score,
        )


# Keep the class spelling used by demo_CLI.py and the official source.
watermark_model = WatermarkModel

