# System One model plan

Status: Phase 4 done. Gate met. The LoRA stays. A plain-model re-score at the last content token is in progress. This file is the handoff for a future agent. Do not skip ahead of the phase gates.

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

A uniform guess over the real option counts scores 0.318 (600 two-way questions, 1,100 four-way, 300 five-way). The readout is 0.058 above that. Mean confidence on the chosen letter is 0.742. That gap is the ECE: the model states about 74% while it is right 37.6% of the time. Temperature is not applied. Fitting it on this file would use the exam as the dial. That fit is Phase 3, on the calibration pile.

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

Status: done. Gate met. Curve: `results/phase2/train.json`. LoRA config: `results/phase2/adapter_config.json`. Weights: Modal volume `phase2-lora`, path `adapter/`. The container is stopped.

Unsloth `FastVisionModel` loaded `unsloth/gemma-4-E4B-it` in 16-bit and attached a rank-16 LoRA. 588 trainable tensors, all under `language_model` attention and MLP. No vision, audio, image, or projector tensor was trainable. One pass, 40820 rows, 5103 steps, batch 8, max length 2048. Letter cross-entropy at the last real token. `SFTTrainer` was not used. The saved file is a PEFT LoRA (`peft_type: LORA`), not a merged base model.

The loss fell. The mean of the logged losses at steps 1, 26, 51, and 76 is 0.975. The mean of the last 100 steps is 0.246. Step 1 was 1.487.

A client left connected for about 21 minutes had its input cancelled, so the epoch was finished in 10-minute chunks that resumed from the volume. Each chunk started a fresh AdamW state. The learning-rate schedule still followed the global step. `adapter_model.safetensors` is 140 MB, over GitHub's 100 MB file limit, so the weights stay on the volume.

Gate: `adapter_config.json` exists, the saved tensors are the adapter, and the loss is lower at the end than at the start. This does not say the exam improved. That check is Phase 4.

## Phase 3 — Temperature

Status: done. Gate met. File: `results/phase3/temperature.json`. Copy on the volume `phase2-lora` at `temperature.json`. Run: https://modal.com/apps/omnaidu42/main/ap-HlBDLGyuMlT99yqMQ10aar The container is stopped. The LoRA was loaded from `adapter/` and was not rewritten.

The fit read `data/phase1/calibration.jsonl` only (4,000 rows: 912 noul, 3,088 choice, none shuffled). Grid 0.50 to 3.00 step 0.05. Chosen `T` is the lowest mean letter cross-entropy. Accuracy at that `T` matches accuracy at `T = 1`.

| type | n | T | accuracy | ECE at 1 | ECE at T |
|---|---|---|---|---|---|
| choice | 3088 | 1.30 | 0.923 | 0.032 | 0.009 |
| noul | 912 | 1.65 | 0.909 | 0.055 | 0.015 |

`score` is missing from this pile, so it has no temperature. Phase 4 uses `T = 1` for score.

These rows are the same sources as training. The accuracy above is not the exam. Phase 0 on typed-decisions was 0.376. Phase 4 is the comparison, with `softmax(logits / T)` and the Phase 0 exam prompt.

Gate: the JSON exists, and it was not fit on the train pile, SNLI, or typed-decisions.

## Phase 4 — Exam

Status: done. Gate met. The LoRA stays. File: `results/phase4/exam.json`. Run: https://modal.com/apps/omnaidu42/main/ap-1wFzOpGmDsVQ5qagerOK3L The container is stopped. The LoRA was loaded from `adapter/` and was not rewritten.

The exam prompt matches Phase 0 (`A. id: text`), checked against the saved Phase 0 example. Tokenizer padding matches Phase 0: `padding_side` is `left`, and the scored position is `mask.sum() - 1`. Temperatures: noul 1.65, choice 1.30, score 1.

| | accuracy | ECE | Brier |
|---|---|---|---|
| Phase 0, plain E4B | 0.376 | 0.366 | 0.905 |
| Phase 4, LoRA, same index | 0.471 | 0.247 | 0.737 |

| type | n | Phase 0 accuracy | Phase 4 accuracy | Phase 0 ECE | Phase 4 ECE |
|---|---|---|---|---|---|
| choice | 600 | 0.288 | 0.518 | 0.384 | 0.195 |
| noul | 600 | 0.538 | 0.540 | 0.294 | 0.151 |
| score | 800 | 0.320 | 0.383 | 0.407 | 0.363 |

Accuracy is up and ECE is down, so the gate says keep the LoRA. The score is under the 73.5% ceiling.

The same forward pass, read at the last real token instead of `mask.sum() - 1`, scores accuracy 0.604, ECE 0.162, Brier 0.550. Those two indexes disagree on 824 of 2,000 questions because the tokenizer left-pads. Phase 0 used `mask.sum() - 1`, so the gate uses that index. The 0.604 figure is not a comparison with Phase 0. Phase 0 was not re-scored at the last real token.

Option order was reversed and the letters were rebound. The chosen option id changed on 1,254 of 2,000 questions (0.627). Slot 0 was the pick on 762 questions (38.1%). The gold label is in slot 0 on 538 (26.9%). Phase 0 picked slot 0 on 958 (47.9%). The first-slot habit is smaller and still present. Flipped-prompt accuracy is 0.476.

SNLI, 2,000 rows never used in training, training prompt, right padding, choice temperature 1.30: accuracy 0.910, ECE 0.013, Brier 0.134. That score is the same three-way label set as MultiNLI. It is not the typed-decisions exam, and it does not enter the keep boolean.

## Suggested next measurement

Not started. Re-score plain E4B on the same 2,000 exam prompts, reading the last real token. Phase 4’s last-real-token accuracy is 0.604, and Phase 0 was read at `mask.sum() - 1` because the tokenizer left-pads. Until the plain model is read at the last real token, 0.604 has no baseline. On that pass, also keep per-type numbers, the slot histogram, and the reversed-option count for the adapter. Do not train. Do not merge. The position habit and the missing score rows are later questions.

## Out of scope for v1

- Merging the LoRA into Gemma.
- Training on Jev’s distill corpus as the main set.
- A Clef-style custom head or a 27B run.
- Embeddings. This model is the judge, not the retriever.
