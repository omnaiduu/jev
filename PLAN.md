# System One model plan

Status: not started. This file is the handoff for a future agent. Do not skip ahead of the phase gates.

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

Load E4B. Do not train. On every exam question, logit-readout the option words and softmax. Save accuracy, ECE, and Brier.

Gate: these three numbers are written to disk. Later phases compare against them.

Exam: `LocalLLaMA/typed-decisions` (400 cases, 2,000 questions). The labels agree with themselves about 73.5% of the time. That is the ceiling, not 95%.

## Phase 1 — Data

Build the train, calibration, and test piles under the rules above. Save them as JSONL. Record source, split, and row counts.

Gate: a manifest lists counts per source and confirms typed-decisions is absent from train and calibration.

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
