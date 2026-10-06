# System One model plan

Status: Phase 1 done. Phases 2–4 not started. This file is the handoff for a future agent. Do not start the next phase until asked. Do not skip ahead of the phase gates.

The reasons for each choice, with the examples from the design questions, are in `LEARNING.md`. Read that before changing the approach.

Goal: a small judge. Given a situation and a closed list of answers, return a percentage for each answer. No generated paragraph.

Model: `google/gemma-4-E4B-it` (Unsloth id `unsloth/gemma-4-E4B-it`). Text layers only. Leave vision and audio layers off. LoRA rank 16. E4B LoRA needs about 17 GB, so a Modal L40S or A10 is enough.

Account: Modal profile `omnaidu42` is already authenticated in this environment. Do not write tokens into the repo.

## Terms

- Logit: the raw score the model assigns to a token before it becomes a percentage.
- Softmax: turns logits into percentages that are positive and sum to 1.
- Logit readout: read the logits of the option words only, then softmax. No token generation.
- LoRA: a small adapter on a frozen model. Save it as its own file. Do not merge it into Gemma.
- Cross-entropy: for a single correct option, `loss = -log(percentage on the correct option)`. If `target` is a spread such as `[0.7, 0.2, 0.1]`, sum `-target_i * log(p_i)` over the options. KL divergence pushes the weights the same way, because it differs from cross-entropy by a term that does not depend on the model.
- Temperature `T`: at inference, `softmax(logits / T)`. `T > 1` makes the model look less sure. `T < 1` makes it look more sure. The winning option does not change.
- ECE: expected calibration error. Bucket test questions by the percentage on the top option. Compare that percentage to the fraction that were actually right. Average the gaps. Lower is better.
- Brier: one score that gets worse when the model is wrong or too sure.

## Row format

```json
{
  "state": "I was charged twice. Please refund.",
  "question": "Which team?",
  "options": ["billing", "tech", "sales"],
  "target": [1.0, 0.0, 0.0]
}
```

`target` is in the same order as `options`. Question types:

| type | options | target |
|---|---|---|
| noul (yes/no) | `["no", "yes"]` | `[0, 1]` or a soft pair |
| choice | the labels, about 20 or fewer | one-hot or soft |
| score | ordered levels | percentages across levels |

## Data rules

Bulk of train rows: rewrite open sets into the row format.

- BoolQ for yes/no
- MultiNLI for entail / neutral / contradict
- Banking77 for intent

Generated rows only for tasks those three do not cover (a refund rule, “does this passage answer the question?”). A script chooses the true label. A larger model writes the text. The writer must not fill `target`.

Shuffle option order on about 30% of train rows and move `target` with the words.

Cut three piles before any training and do not reshuffle them later:

| pile | size | use |
|---|---|---|
| train | about 40k | updates the LoRA |
| calibration | 2–5k | fit `T` only |
| test | public exam | never in train or calibration |

Do not train on `LocalLLaMA/typed-decisions`. Do not use `jev-distill-corpus` as the main train set. Those labels are Jev’s own answers. A distill run is an optional side experiment after the main LoRA exists.

## Phase 0 — Baseline

Status: done. Gate met.

Load E4B. Do not train. On every exam question, bind each option to one letter (`A`, `B`, `C`, ...) in dataset order, then softmax only those letter logits. No temperature (`T = 1`). Thinking off. One forward pass per question.

Gate: accuracy, ECE, and Brier are in `results/phase0/baseline.json`. Per-question rows are in `results/phase0/predictions.jsonl` (2,000 lines). Later phases compare against these three numbers, using this same letter readout.

Exam: `LocalLLaMA/typed-decisions`, config `all`, split `test` (400 cases, 2,000 questions). The train split of that dataset was not scored and must stay out of training. The labels agree with themselves about 73.5% of the time. That is the ceiling, not 95%.

Run: Modal app `phase0-gemma-e4b-baseline` on an L40S. https://modal.com/apps/omnaidu42/main/ap-3HrXOJFdqf2uVCy827m8lw

Latency, separate from the gate, is in `results/phase0/latency.json`. On an L40S, one question: prefill about 65–78 ms, decoding the letter about 130–145 ms, a forced 32-token paragraph about 2 seconds. The container was single-use and is stopped. See `LEARNING.md`.

| slice | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| all | 2000 | 0.376 | 0.366 | 0.905 |
| choice | 600 | 0.288 | 0.384 | 0.971 |
| noul | 600 | 0.538 | 0.294 | 0.714 |
| score | 800 | 0.320 | 0.407 | 0.999 |
| agent_trace_observability | 500 | 0.394 | 0.318 | 0.873 |
| customer_service | 500 | 0.288 | 0.490 | 1.101 |
| invoice_processing | 500 | 0.416 | 0.318 | 0.838 |
| security_incidents | 500 | 0.406 | 0.339 | 0.809 |

Exact floats are in the JSON. Rounded here to three decimals.

A uniform guess over the real option counts scores 0.318 (600 two-way questions, 1,100 four-way, 300 five-way). The readout is 0.058 above that. Mean confidence on the chosen letter is 0.742. That gap is the ECE: the model states about 74% while it is right 37.6% of the time. Temperature is not applied. Fitting it on this file would use the exam as the dial. That is Phase 3, on a calibration pile that does not exist yet.

Position bias: letter `A` (slot 0) was the pick on 958 / 2,000 questions (47.9%). The gold label is slot 0 on 538 / 2,000 (26.9%). Picked vs gold by slot: 0 is 958 vs 538, 1 is 454 vs 585, 2 is 289 vs 493, 3 is 263 vs 330, 4 is 36 vs 54. Phase 4 must flip option order and score with the same letters. The train shuffle exists to fight this bias.

Two schema facts the scorer had to handle:

- 200 noul questions omit `criteria` (`invoice_processing` duplicate ×100, `security_incidents` credential_compromise ×100). Gold is still `false` or `true`. The prompt uses "The statement is not true." and "The statement is true."
- Some option ids are more than one Gemma token (`human_review` is three, `harmful` is two). A softmax has one slot per option, so those ids cannot be the readout. Letters are. Do not switch Phase 4 to raw option-token logits without recording it as a different metric.

## Phase 1 — Data

Status: done. Gate met. Files: `data/phase1/train.jsonl`, `data/phase1/calibration.jsonl`, `data/phase1/held_out.jsonl`, `data/phase1/manifest.json`.

| pile | rows | role |
|---|---|---|
| train | 40820 | LoRA updates. 30% of these rows have shuffled options (12246). |
| calibration | 4000 | fit `T` only. Not shuffled. |
| held out | 2000 | SNLI only. Absent from train and calibration. Phase 4 scores this. |
| test | public exam | `LocalLLaMA/typed-decisions` config `all` split `test`. Not copied into these files. |

Train by source: Banking77 9101, BoolQ 8586, MultiNLI 20947, refund_rule 1457, passage_answer 729.

The refund emails and support notes were written by `google/gemma-4-12B-it`. The script wrote `target`. The writer container was single-use and is stopped. SNLI is the source left entirely out of training. typed-decisions is absent from train and calibration.

## Phase 2 — Train

Unsloth loads E4B and attaches the LoRA (rank 16, language / attention / MLP only, vision and audio off). Do not use `SFTTrainer`. The PyTorch loss is the cross-entropy above. One pass over the train pile. Save the LoRA. Do not merge.

Gate: a checkpoint file exists and a short smoke batch shows the loss falling.

## Phase 3 — Temperature

Freeze the LoRA. Run it on the calibration pile. Fit one `T` per question type by minimizing cross-entropy on that pile (L-BFGS, or a small grid such as 0.5, 0.8, 1.0, 1.5, 2.0). Save `T` beside the LoRA.

Gate: a JSON file of temperatures exists. It was not fit on the train pile or the test pile.

## Phase 4 — Exam

Score the LoRA on typed-decisions with `softmax(logits / T)`. Report accuracy, ECE, and Brier next to the Phase 0 numbers. Also:

- Flip option order and count how often the chosen option changes.
- Score one source that was left entirely out of training.

Keep the LoRA only if accuracy is higher than Phase 0 and ECE is lower. A score a little above the 73.5% ceiling means the model fit the teacher’s quirks.

## Out of scope for v1

- Merging the LoRA into Gemma.
- Training on Jev’s distill corpus as the main set.
- A Clef-style custom head or a 27B run.
- Embeddings. This model is the judge, not the retriever.
