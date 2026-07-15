# :sweat_drops: [Linguistic-Based Watermarking for Text Authentication](https://arxiv.org/abs/2305.08883)

Official implementation of the watermark injection and detection algorithms presented in the [paper](https://arxiv.org/abs/2305.08883):

"Linguistic-Based Watermarking for Text Authentication" by _Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, and Nenghai Yu_.  

## Reproduce the core method in the current dlm environment

The original modules pin an old dependency stack and import Chinese and English dependencies together. `models/watermark_compat.py` provides an English-only implementation for the current high-version environment while keeping the paper pipeline.

The following paper-aligned components are active:

1. The official NLTK Penn Treebank POS whitelist and English stop-word filter.
2. Partially masked BERT embeddings for contextual candidate generation.
3. Context similarity from the last eight BERT hidden layers.
4. `glove-wiki-gigaword-100` for global word similarity.
5. Two-sentence processing units for embedding and detection.
6. SHA-256 bit encoding and the one-sided z-test in fast or precise mode.
7. Paper defaults: `K=32`, `lambda=0.83`, `tau_sent=0.8`, and `tau_word=0.8`.

The paper model paths are now active:

* `/data/llm/bert-base-cased` for candidate generation and contextual similarity;
* `/data/llm/roberta-large-mnli` for sentence-level MNLI entailment probability;
* `/data/llm/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz` for global word similarity.

The NLTK stopwords and `averaged_perceptron_tagger_eng` resources are installed under `/data/wangzhuoshang/nltk_data`. The sentence score follows the official executable repository, which uses the MNLI entailment probability at label 2. This is slightly more implementation-specific than the cosine-similarity notation in the paper. Exact paper result tables are not targeted.

Run an end-to-end embedding and fast-detection check on GPU 0:

```sh
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python run_watermark.py roundtrip --device cuda \
  --text "A sufficiently long English passage to watermark and detect ..."
```

Embed text only:

```sh
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python run_watermark.py embed --device cuda --input-file input.txt
```

Fast detection does not load semantic models and can run on CPU:

```sh
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python run_watermark.py detect --detector fast --input-file watermarked.txt
```

Precise detection regenerates and filters synonym candidates, so it needs the models and is slower:

```sh
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python run_watermark.py detect --detector precise --device cuda \
  --input-file watermarked.txt
```

## Requirements
- Python 3.9
- check requirements.txt
```sh
pip install -r requirements.txt
pip install git+https://github.com/JunnYu/WoBERT_pytorch.git  # Chinese word-level BERT model
python -m spacy download en_core_web_sm
```
- For Chinese, please download the [pre-trained Chinese word vectors](https://drive.google.com/file/d/1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR/view) and place it in the root directory.

## Repo contents

The watermark injection and detection modules are located in the `models` directory. `watermark_original.py` implements the iterative algorithms as described in the paper. `watermark_faster.py` introduces batch processing to speed up the watermark injection algorithm and the precise detection algorithm.

We provide two demonstrations, `demo_CLI.py` and `demo_gradio.py`, which correspond to command-line interaction and graphical interface interaction respectively.

## Demo Usage
> Click on the GIFs to enlarge them for a better experience.
### Graphical User Interface
```sh
$ python demo_gradio.py --language English --tau_word 0.8 --lamda 0.83
```
<p align="center">
  <img src="images/en_gradio.gif" />
</p>

```sh
$ python demo_gradio.py --language Chinese --tau_word 0.75 --lamda 0.83
```
<p align="center">
  <img src="images/cn_gradio.gif" />
</p>

### Command Line Interface
```sh
$ python demo_CLI.py --language English --tau_word 0.8 --lamda 0.83
```
<p align="center">
  <img src="images/eng_cli.gif" />
</p>

```sh
$ python demo_CLI.py --language Chinese --tau_word 0.75 --lamda 0.83
```

<p align="center">
  <img src="images/cn_cli.gif" />
</p>


