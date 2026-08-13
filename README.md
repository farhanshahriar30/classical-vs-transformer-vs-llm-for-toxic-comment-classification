# Toxic Comment Classification: Classical ML vs. Fine-Tuned Transformers vs. LLM Prompting

A controlled three-paradigm comparison for multi-label toxic comment detection. TF-IDF with logistic regression, a fine-tuned RoBERTa-base, and zero-shot / few-shot LLM prompting are evaluated on the same task, the same splits, and the same metrics, so the differences reflect the modelling approach rather than the experimental setup.

**Headline result:** fine-tuned RoBERTa wins on every aggregate metric, and its margin is widest exactly where it matters most, on the rare labels that aggregate scores tend to hide.

> **Content note:** this project uses the Jigsaw toxic comment dataset, which contains offensive and abusive language. Examples in the notebooks and outputs reflect that.

---

## Why this project

Most write-ups pick one paradigm and report its best number. That makes it hard to answer the question a practitioner actually has: given labelled data, compute, and a deadline, which approach should I reach for?

This project answers that by holding the evaluation protocol fixed and varying only the method. It also treats evaluation as a first-class design problem rather than a final step, because the task is both **multi-label** (a comment can be toxic, obscene, and insulting at once) and **heavily imbalanced** (`threat` and `identity_hate` are rare), and both properties break naive evaluation.

---

## Task

Six binary labels, any combination of which may apply to a single comment:

`toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

This is six related binary problems learned jointly over the same input, not a six-way choice.

**Dataset:** Jigsaw Toxic Comment Classification Challenge (~160k comments).

**Splits:** `train.csv` was split 70 / 15 / 15 into 111,699 train, 23,936 validation, and 23,936 test, stratified approximately on whether a comment carries any active label. Random seed 42 throughout.

The provided `test.csv` and `test_labels.csv` were **not** used as the primary evaluation set, because `test_labels.csv` contains `-1` entries for comments excluded from scoring. Using a controlled split of the fully labelled training file gives a cleaner and more consistent comparison.

---

## Results

### Aggregate performance

| Model | Evaluation set | Micro F1 | Macro F1 | Mean PR-AUC |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | Full held-out test | 0.731 | 0.605 | 0.634 |
| **RoBERTa fine-tuned** | Full held-out test | **0.782** | **0.686** | **0.728** |
| LLM zero-shot | 600-comment subset | 0.743 | 0.578 | n/a |
| LLM few-shot (k=4, retrieved) | 600-comment subset | 0.752 | 0.611 | n/a |

**Read this table carefully.** The two LLM rows are scored on a 600-comment subset of the held-out test split, not the full 23,936, because API cost and latency made full-set inference impractical. They are directionally informative but not strictly comparable to the supervised rows. PR-AUC is absent for the LLM rows because prompting returns hard labels rather than calibrated probabilities.

### Per-label F1, supervised models

| Label | TF-IDF + LR | RoBERTa | Delta |
|---|---:|---:|---:|
| toxic | 0.766 | 0.822 | +0.056 |
| obscene | 0.795 | 0.836 | +0.041 |
| insult | 0.709 | 0.770 | +0.061 |
| severe_toxic | 0.490 | 0.535 | +0.045 |
| threat | 0.421 | 0.587 | **+0.166** |
| identity_hate | 0.448 | 0.566 | **+0.118** |

The gap between the two models is roughly 0.05 on the frequent labels and roughly 0.12 to 0.17 on the rare ones. Contextual representations help most where lexical cues are weak and training examples are scarce, which is also where a moderation system is most likely to fail in production.

### Zero-shot vs. few-shot prompting

Few-shot improved macro F1 more than micro F1 (0.578 to 0.611 versus 0.743 to 0.752), meaning the retrieved demonstrations helped spread performance across labels rather than just reinforcing the common ones.

The gain was not uniform. `severe_toxic` rose sharply from 0.130 to 0.333 and `insult` from 0.670 to 0.703, while `obscene` and `identity_hate` declined slightly and `threat` was essentially unchanged. Because demonstrations were retrieved by TF-IDF cosine similarity, some were lexically close to the target comment without being informative about its label structure. Retrieval quality, not the presence of examples, is what determines whether few-shot helps.

---

## Method

### 1. Classical baseline

TF-IDF with one-vs-rest logistic regression.

```
TfidfVectorizer(ngram_range=(1,2), max_features=30000, min_df=2,
                sublinear_tf=True, lowercase=True)
LogisticRegression(solver="liblinear", max_iter=1000)
```

One model per label, producing independent probability estimates across all six categories.

### 2. Transformer fine-tuning

`roberta-base` with a multi-label classification head.

- Max sequence length 256, truncation and padding to fixed length
- Encoder, dropout, then a linear layer with six outputs
- `BCEWithLogitsLoss`, one logit per label, no softmax
- AdamW, learning rate 2e-5, batch size 8, 3 epochs
- Validation loss monitored each epoch, best checkpoint saved

### 3. LLM prompting

OpenAI API, temperature 0 for deterministic output, evaluated on a 600-comment subset of the held-out test split balanced to include both labelled and unlabelled comments.

The prompt supplies the task instruction, the six valid labels, and formatting rules, and requires a JSON array containing only labels from the allowed set, with an empty array when nothing applies. The few-shot variant adds four demonstrations retrieved from the **training split only** by TF-IDF cosine similarity against the target comment, so no evaluation leakage is introduced.

---

## Evaluation design

Three decisions do most of the work here.

**Per-label threshold tuning, not a fixed 0.5.** Thresholds were selected independently for each label by maximising F1 on the validation split, then applied once to the held-out test set. A default threshold is close to indefensible for rare labels, where the optimal operating point sits nowhere near 0.5.

- Classical: per-label F1 search on validation
- RoBERTa: sigmoid probabilities, threshold swept from 0.10 to 0.90 in steps of 0.01

**Macro F1 and per-label F1 alongside micro F1.** Micro F1 is dominated by the frequent labels. A model can look strong on micro F1 while failing almost completely on `threat`. Macro F1 weights every label equally and is what actually exposes that.

**PR-AUC rather than ROC-AUC as the ranking metric.** Under heavy imbalance, precision-recall analysis is more informative than ROC, which can look optimistic when negatives dominate.

---

## Error analysis

Failure modes were consistent across all three paradigms, differing in severity rather than in kind.

**Rare labels are the bottleneck.** `severe_toxic`, `threat`, and `identity_hate` scored lowest everywhere. A common pattern: the model correctly flags a comment as generally toxic but misses the more specific label capturing its severity or target.

**Label overlap produces partial credit failures.** Broad categories such as `toxic` function as umbrella labels, while `severe_toxic` and `identity_hate` require finer distinctions. Models frequently recognised that a comment was harmful but predicted an incomplete label set. In multi-label classification the full label set is the prediction, so partial correctness is still an error.

**Prompting has a failure mode the supervised models do not.** Zero-shot underpredicts rare labels, collapsing multi-label cases to the most obvious category. Few-shot fixes some of this but introduces a dependency on retrieval quality, which is why its per-label gains were mixed.

---

## What I would take from this

**Fine-tune when you have labels and compute.** RoBERTa won on every aggregate metric and was the most consistent on the hard labels.

**Do not dismiss the classical baseline.** TF-IDF with logistic regression trains in seconds, runs cheaply, is fully interpretable, and landed within about 0.05 micro F1 of a fine-tuned transformer. For a first-pass screen or a latency-constrained tier, that trade is often correct.

**Treat prompting as a cold-start tool.** Zero-shot classification with no task-specific training is genuinely useful when labels do not exist yet. It was less stable than supervised fine-tuning here and more sensitive to prompt and demonstration quality, so it reads as a supplement rather than a replacement.

---

## Repository structure

```
.
├── src/
│   ├── data_utils.py            # loading, splitting, stratification
│   ├── classical_model.py       # TF-IDF + one-vs-rest logistic regression
│   ├── transformer_model.py     # RoBERTa fine-tuning and inference
│   ├── llm_prompting.py         # zero-shot and few-shot prompt construction
│   └── metrics.py               # micro/macro/per-label F1, PR-AUC, thresholds
├── notebooks/
│   ├── 01_data_and_classical.ipynb   # splits, EDA, classical baseline
│   ├── 02_transformer.ipynb          # RoBERTa fine-tuning
│   └── 03_llm_prompting.ipynb        # zero-shot and few-shot evaluation
├── data/                        # Jigsaw CSVs and generated splits (not tracked)
├── figures/
├── results/                     # metric tables, thresholds, predictions
├── requirements.txt
├── LICENSE
└── README.md
```

Key outputs:

| File | Contents |
|---|---|
| `results/final_model_comparison.csv` | Headline results table, all four approaches |
| `results/classical_overall_metrics.csv` | TF-IDF baseline aggregate scores |
| `results/roberta_overall_metrics_v2.csv` | Fine-tuned RoBERTa aggregate scores |
| `results/*_per_label_metrics*.csv` | Per-label F1 for every approach |
| `results/llm_zero_vs_few_shot_comparison.csv` | Prompting comparison |
| `results/classical_best_thresholds.csv`, `results/roberta_best_thresholds_v2.csv` | Selected per-label thresholds |
| `results/paper_example_predictions_table.csv` | Side-by-side predictions across all four methods |
| `figures/roberta_training_history_v2.png` | Training and validation loss by epoch |
| `figures/label_frequency_train.png` | Label distribution, showing the imbalance |

Files suffixed `_v2` are the final RoBERTa run and the source of the reported numbers.

## Reproducing

```bash
git clone <REPO_URL>
cd <REPO_NAME>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Jigsaw dataset from Kaggle and place `train.csv`, `test.csv`, and `test_labels.csv` in `data/`. The train/validation/test splits are generated by the first notebook and written back to `data/` as `train_split.csv`, `val_split.csv`, and `test_split.csv`.

Run the notebooks in order:

1. `01_data_and_classical.ipynb` builds the splits and trains the classical baseline
2. `02_transformer.ipynb` fine-tunes RoBERTa
3. `03_llm_prompting.ipynb` runs zero-shot and few-shot evaluation

Seed 42 is fixed for splitting, sampling, and training.

The LLM notebook requires an OpenAI API key in a local `.env` file:

```
OPENAI_API_KEY=your-key-here
```

Fine-tuning RoBERTa needs a GPU. It ran in roughly 150 minutes for 3 epochs at batch size 8 in this setup. The classical baseline and all evaluation run on CPU.

---

## Limitations and next steps

- **The LLM comparison is on a subset.** 600 comments versus 23,936. Directional, not definitive.
- **No fairness evaluation.** Toxicity classifiers are known to associate identity terms with toxicity and produce systematically different behaviour across identity groups. Subgroup metrics are the most important missing piece here and the first thing I would add.
- **Calibration is discussed but not measured.** Threshold tuning was done on F1, not on calibration quality. In a moderation setting where scores drive ranking and review queues, reliability diagrams and temperature scaling would matter.
- **Few-shot retrieval is purely lexical.** TF-IDF similarity finds lexically close examples, not label-structurally informative ones. Embedding-based or label-aware retrieval is the obvious improvement.
- **One transformer, one LLM.** No domain-adapted encoder such as HateBERT, and no comparison across model providers.

---

## References

Key work this project builds on:

- Devlin et al. (2019), BERT
- Liu et al. (2019), RoBERTa
- Brown et al. (2020), few-shot prompting
- Saito and Rehmsmeier (2015), precision-recall over ROC under imbalance
- Borkan et al. (2019), unintended bias in toxicity classification
- Edwards and Camacho-Collados (2024), whether in-context learning is enough for text classification

Full reference list is in the accompanying paper.
