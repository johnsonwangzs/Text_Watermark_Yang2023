"""Non-interactive CLI for the dlm-compatible watermark implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.watermark_compat import (
    DEFAULT_GLOVE_PATH,
    DEFAULT_MLM_PATH,
    DEFAULT_SENTENCE_MODEL_PATH,
    WatermarkModel,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed or detect the Yang et al. (2023) lexical watermark."
    )
    parser.add_argument("mode", choices=["embed", "detect", "roundtrip"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to process")
    source.add_argument("--input-file", type=Path, help="UTF-8 text file to process")
    parser.add_argument("--detector", choices=["fast", "precise"], default="fast")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tau-word", type=float, default=0.8)
    parser.add_argument("--tau-sent", type=float, default=0.8)
    parser.add_argument("--lamda", type=float, default=0.83)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--dropout-prob", type=float, default=0.3)
    parser.add_argument("--mlm-path", default=DEFAULT_MLM_PATH)
    parser.add_argument("--sentence-model-path", default=DEFAULT_SENTENCE_MODEL_PATH)
    parser.add_argument("--glove-path", default=DEFAULT_GLOVE_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser


def read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return args.input_file.read_text(encoding="utf-8")


def make_model(args: argparse.Namespace, *, semantic: bool) -> WatermarkModel:
    return WatermarkModel(
        tau_word=args.tau_word,
        tau_sent=args.tau_sent,
        lamda=args.lamda,
        top_k=args.top_k,
        dropout_prob=args.dropout_prob,
        mlm_path=args.mlm_path,
        sentence_model_path=args.sentence_model_path,
        glove_path=args.glove_path,
        device=args.device,
        load_semantic_models=semantic,
    )


def main() -> None:
    args = build_parser().parse_args()
    text = read_text(args).strip()
    precise = args.detector == "precise"

    if args.mode == "detect":
        model = make_model(args, semantic=precise)
        result = model.detect(text, alpha=args.alpha, precise=precise)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return

    model = make_model(args, semantic=True)
    embedding = model.embed_with_report(text)
    if args.mode == "embed":
        print(embedding.text)
        return

    before = model.detect(text, alpha=args.alpha, precise=precise)
    after = model.detect(embedding.text, alpha=args.alpha, precise=precise)
    report = {
        "backends": model.backend_info,
        "original_text": text,
        "watermarked_text": embedding.text,
        "num_substitutions": embedding.num_substitutions,
        "substitutions": [substitution.__dict__ for substitution in embedding.substitutions],
        "detection_before": before.to_dict(),
        "detection_after": after.to_dict(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

