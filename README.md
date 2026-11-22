# Gemini Fine-Tuning for Medical Diagnosis (INFO 7374 final project)

Final project for Advanced Techniques with LLMs (INFO 7374), Northeastern University.

## What it is

Fine-tuned Gemini 1.5 Flash on 7,652 medical case → diagnosis pairs, then built
a Python evaluation harness to compare the base model against the fine-tuned
model on 40 held-out test cases (20 new samples + 20 from the fine-tuning
distribution).

## Files

- `FineTuning_Data.csv` — 7,652 Query/Response pairs (patient case description
  → diagnosis) used for fine-tuning.
- `main.py` — evaluation harness: runs the base model (gemini-1.5-flash) on the
  test set and scores accuracy.
- `main2.py` — same harness, pointed at the fine-tuned model
  (`tunedModels/finetuningdata2-sk1hvsk04ps2`).
- `Test_Data.xlsx` — 40 test cases with correct diagnoses and per-model results.

## Method

- Scoring: keyword overlap between the model's first-sentence diagnosis and the
  correct diagnosis (stopwords removed); ≥50% overlap counts as correct.
- Two fine-tuning settings, batch size 8 in both runs:
  - Setting 1: learning-rate multiplier 0.1, 10 epochs
  - Setting 2: learning-rate multiplier 2, 5 epochs

## Results

Fine-tuning did not improve accuracy:

| setting | new samples (n=20) | in-distribution (n=20) |
| before fine-tuning | 30% (6/20) | 55% (11/20) |
| Setting 1 (LR 0.1, 10 ep) | 35% (7/20) | 35% (7/20) |
| Setting 2 (LR 2, 5 ep) | 25% (5/20) | 30% (6/20) |

The honest read: the fine-tuned model got slightly better on new cases at the
gentler setting and worse everywhere else — a useful negative result, and the
reason the evaluation harness exists.

## Setup

Set your key as an environment variable (never hardcode it):

```
export GEMINI_API_KEY="your-key-here"
python3 main.py
```
