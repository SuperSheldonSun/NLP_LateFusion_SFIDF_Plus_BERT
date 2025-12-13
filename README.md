````markdown
# Personalized News Recommendation on MIND-small
**TF-IDF / SF-IDF / SF-IDF+ / Sentence-BERT / Late Fusion**  
**Unified Evaluation: Hit@K, nDCG@K, MRR**

This repo implements a modular retrieval-based news recommender on **MIND-small**.  
We compare **symbolic sparse representations** (TF-IDF, SF-IDF, SF-IDF+) with a **dense semantic retriever** (Sentence-BERT), and a **Late Fusion** strategy that interpolates normalized scores.

All ranking outputs share the same TSV format:
```text
user_id \t session_id \t doc_id \t score \t label
````

---

## Contents

* [Key Features](#key-features)
* [Dataset](#dataset)
* [Method Overview](#method-overview)
* [Environment & Installation](#environment--installation)
* [Project Layout](#project-layout)
* [Configuration](#configuration)
* [End-to-end Pipeline](#end-to-end-pipeline)
* [Evaluation](#evaluation)
* [Paper Results (Dev)](#paper-results-dev)
* [Late Fusion Grid Search](#late-fusion-grid-search)
* [Notes & Known Limitations](#notes--known-limitations)
* [Citation](#citation)
* [License](#license)

---

## Key Features

Supported ranking systems:

* **TF-IDF** baseline
* **SF-IDF** (WordNet synsets only)
* **SF-IDF+** (synsets + NER entities)
* **BERT-only** (Sentence-BERT)
* **Late Fusion** (BERT + SF-IDF+)

Unified evaluation toolkit:

* Hit@K
* nDCG@K
* MRR

---

## Dataset

We run experiments on **MIND-small**, a lightweight subset of the Microsoft News Dataset (MIND).

**MIND-small highlights (as used in our paper/write-up):**

* Training set: ~156k impression logs
* Dev/validation set: ~73k impression logs
* News content: 65k+ news articles (title, abstract, category, subcategory)

### Expected folder structure

Place `news.tsv` and `behaviors.tsv` under:

```
data/
  MINDsmall_train/
    news.tsv
    behaviors.tsv
  MINDsmall_dev/
    news.tsv
    behaviors.tsv
```

---

## Method Overview

### Document text

For each news article `d`, we build the document text by concatenating:

* **Title + Abstract**

### Preprocessing (symbolic pipeline)

Before SF-IDF/SF-IDF+, we apply:

1. **Tokenization & filtering** (drop non-alphanumeric tokens)
2. **POS tagging**
3. **Lemmatization** (WordNet lemmatizer with POS)

### Sparse encoders

#### 1) TF-IDF

Standard TF-IDF on the corpus.

#### 2) SF-IDF (synsets only)

We map tokens to **WordNet synsets** to reduce synonymy mismatch.
This creates a sparse vector in synset-feature space.

#### 3) SF-IDF+ (synsets + entities)

SF-IDF+ adds a parallel **Named Entity** channel (via spaCy NER, with optional fallback rules).
We combine synset and entity vectors with weight **α**:

* `v_sfidf_plus = α * v_synset + (1 - α) * v_entity`

Default in the paper main run: **α = 0.5**.

### Dense encoder (Sentence-BERT)

We use:

* `SentenceTransformer("all-MiniLM-L6-v2")`
* Input: concatenated **title + abstract**
* Output: **384-dim** dense embedding
* We apply **L2 normalization** so dot-product equals cosine similarity.

### User profiling (centroid)

For user `u`, with clicked history `H_u = {d1, d2, ..., dk}`:

* `v_user = mean( v_doc(d) for d in H_u )`

Computed separately for:

* `mode ∈ { sfidf_plus, bert }`

If a user has no history, we use a zero vector.

### Scoring & ranking

For each session (impression) with candidate document `d_c`:

* Dense score: `s_bert = v_user_bert · v_doc_bert(d_c)`
* Symbolic score: `s_sym  = v_user_sfidf_plus · v_doc_sfidf_plus(d_c)`

### Late Fusion (with session-level Min-Max normalization)

Dense and sparse scores have different distributions. We normalize **per session**:

* `s' = (s - min(S)) / (max(S) - min(S))`
  where `S` is the set of candidate scores in the same session.

Final fused score with weight **λ**:

* `Score(u, d_c) = λ * s'_bert + (1 - λ) * s'_sym`

Default in the paper main run: **λ = 0.5**.

---

## Environment & Installation

### Python

* Python **3.9+** recommended

### Install dependencies

```bash
pip install numpy scipy scikit-learn sentence-transformers nltk spacy pyyaml tqdm
python -m spacy download en_core_web_sm
```

### Download NLTK resources (run once)

```python
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
```

---

## Project Layout

Directory layout assumption:

* Project root: `NLP_Project`
* Data: `data/MINDsmall_train`, `data/MINDsmall_dev`

A typical repo layout:

```
NLP_Project/
  scripts/
    prepare_data.py
    build_sfidf.py
    build_bert.py
    run_rank.py
    grid_search_late_fusion.py
  eval.py
  config.yaml
  data/
  cached/
  vectors/
  outputs/
```

---

## Configuration

Main configuration file: `config.yaml`.

Key fields:

* `data.root_dir`: project root (default `.`)
* `paths.cached_dir`: cache directory (default `cached`)
* `paths.vectors_dir`: SF-IDF / BERT vectors (default `vectors`)
* `paths.outputs_dir`: ranking outputs (default `outputs`)
* `sfidf.alpha`: α for SF-IDF+ (synset/entity mix)
* `fusion.lambda_fusion`: λ for Late Fusion
* `eval.k_list`: K values for Hit@K / nDCG@K (e.g., `[5, 10]`)

All scripts accept:

* `--config config.yaml`
  CLI args override config values.

---

## End-to-end Pipeline

Assume your current working directory is `NLP_Project`.

### 1) Optional preprocessing & caching

```bash
python scripts/prepare_data.py --config config.yaml
```

* Reads `news.tsv` and `behaviors.tsv`
* Builds session-based instances and caches processed objects under `cached/`

### 2) Build SF-IDF / SF-IDF+ vectors

```bash
python scripts/build_sfidf.py --config config.yaml
```

What it does:

* Iterate over every news article in `train + dev`
* Extract WordNet synsets + Named Entities
* Build corpus DF stats (synset + entity)
* Save vectors (example outputs):

  * `vectors/sfidf_vectors.pkl`
  * `vectors/sfidf_entity_vectors.pkl`
  * `vectors/sfidf_plus_vectors.pkl`

### 3) Build BERT vectors

```bash
python scripts/build_bert.py --config config.yaml
```

* Uses `all-MiniLM-L6-v2`
* Encodes each article (title + abstract)
* L2-normalizes and saves:

  * `vectors/bert_vectors.pkl`

### 4) Run ranking for different modes

Entry point: `scripts/run_rank.py`

Arguments:

* `--mode`: `tfidf`, `sfidf`, `sfidf_plus`, `bert`, `late_fusion`
* `--split`: `train` or `dev`

Examples:

**TF-IDF**

```bash
python scripts/run_rank.py --config config.yaml --mode tfidf --split dev
```

**SF-IDF**

```bash
python scripts/run_rank.py --config config.yaml --mode sfidf --split dev
```

**SF-IDF+**

```bash
python scripts/run_rank.py --config config.yaml --mode sfidf_plus --split dev
```

**BERT-only**

```bash
python scripts/run_rank.py --config config.yaml --mode bert --split dev
```

**Late Fusion**

```bash
python scripts/run_rank.py --config config.yaml --mode late_fusion --split dev --lambda_fusion 0.5
```

Each run writes a TSV under `outputs/`, e.g.:

* `outputs/tfidf_dev_rank.tsv`
* `outputs/sfidf_dev_rank.tsv`
* `outputs/sfidf_plus_dev_rank.tsv`
* `outputs/bert_dev_rank.tsv`
* `outputs/late_fusion_dev_rank.tsv`

TSV format:

```text
user_id \t session_id \t doc_id \t score \t label
```

---

## Evaluation

Use `eval.py` to score any ranking file.

Example (TF-IDF on dev):

```bash
python eval.py --config config.yaml \
  --input outputs/tfidf_dev_rank.tsv \
  --mode tfidf \
  --output_json outputs/metrics_tfidf_dev.json
```

Other modes:

```bash
python eval.py --config config.yaml --input outputs/sfidf_dev_rank.tsv       --mode sfidf       --output_json outputs/metrics_sfidf_dev.json
python eval.py --config config.yaml --input outputs/sfidf_plus_dev_rank.tsv  --mode sfidf_plus  --output_json outputs/metrics_sfidf_plus_dev.json
python eval.py --config config.yaml --input outputs/bert_dev_rank.tsv        --mode bert        --output_json outputs/metrics_bert_dev.json
python eval.py --config config.yaml --input outputs/late_fusion_dev_rank.tsv --mode late_fusion --output_json outputs/metrics_late_fusion_dev.json
```

Console output example:

```text
Results for file: outputs/tfidf_dev_rank.tsv
Mode: tfidf
Hit@5: 0.xxx
Hit@10: 0.xxx
nDCG@5: 0.xxx
nDCG@10: 0.xxx
MRR: 0.xxx
```

---

## Paper Results (Dev)

Main run settings:

* SF-IDF+ α = 0.5
* Late Fusion λ = 0.5

| Model       | Hit@5  | Hit@10 | nDCG@5 | nDCG@10 | MRR    |
| ----------- | ------ | ------ | ------ | ------- | ------ |
| TF-IDF      | 0.4715 | 0.6534 | 0.2837 | 0.3430  | 0.3062 |
| SF-IDF      | 0.4658 | 0.6506 | 0.2785 | 0.3391  | 0.3031 |
| SF-IDF+     | 0.4691 | 0.6536 | 0.2791 | 0.3397  | 0.3010 |
| BERT-only   | 0.5439 | 0.7185 | 0.3344 | 0.3940  | 0.3514 |
| Late Fusion | 0.5262 | 0.7076 | 0.3168 | 0.3783  | 0.3334 |

---

## Late Fusion Grid Search

We grid search over:

* **α**: synset/entity weight in SF-IDF+
* **λ**: fusion weight (BERT vs symbolic)

Default grid search (if baked in):

```bash
python scripts/grid_search_late_fusion.py
```

Override settings (example):

```bash
python scripts/grid_search_late_fusion.py \
  --split dev \
  --alpha_values 0.3 0.5 0.7 \
  --lambda_values 0.5 0.7 \
  --results_json outputs/late_fusion_grid/grid_results_dev.json
```

**Dev grid search snapshot (selected runs from the paper):**

| α   | λ   | Hit@10 | Hit@5  | MRR    | nDCG@10 | nDCG@5 |
| --- | --- | ------ | ------ | ------ | ------- | ------ |
| 0.5 | 0.7 | 0.7167 | 0.5414 | 0.3467 | 0.3903  | 0.3305 |
| 0.3 | 0.7 | 0.7163 | 0.5423 | 0.3450 | 0.3896  | 0.3304 |
| 0.7 | 0.7 | 0.7151 | 0.5406 | 0.3467 | 0.3896  | 0.3299 |
| 0.5 | 0.5 | 0.7076 | 0.5263 | 0.3333 | 0.3783  | 0.3168 |

Observation: performance improves as **λ increases** (favoring BERT).

---

## Notes & Known Limitations

* **WordNet synset mapping is noisy** without robust word-sense disambiguation, which can hurt SF-IDF.
* Adding entities (SF-IDF+) helps recover some performance, but synset noise can still dominate.
* **BERT-only** is strong even in a zero-shot setup (no fine-tuning on MIND).
* Late Fusion may underperform BERT-only due to:

  * score distribution mismatch (even with session-level min-max)
  * centroid averaging being more “natural” for dense vectors than sparse ones
  * representation granularity mismatch (sentence-level vs token-level)

---

## Citation

If you use this repository in a course project or research, please cite your project paper and the MIND dataset paper.

Example BibTeX (edit fields as needed):

```bibtex
@misc{mindsmall_latefusion_2025,
  title  = {Evaluating Sparse Symbolic Signals and Sentence-BERT for Personalized News Recommendation on MIND-small},
  author = {Yibo Sun and Leo Liu and George Guo},
  year   = {2025}
}
```

---

## License

MIT.

```
```
