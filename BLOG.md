# Blog notes

Source notes for the post. The post is not written here. Phase 4 has not been run, so the exam comparison is still open. The public bar until then is the Phase 0 table.

## Working titles

- A judge that returns a percentage list
- One forward pass, then softmax over the letters
- What plain Gemma 4 E4B does when the answers are closed

## The spine

Use these as the through-line. Each one is something the runs actually showed.

1. The product is a percentage for each allowed answer. Code can branch on it. The model stops before it writes a paragraph.
2. The percentage comes from one forward pass. Softmax over the letter logits. No generated sentence.
3. Some option ids are more than one token, so the softmax slot is a letter. The words stay in the prompt.
4. Plain E4B scores 0.376 on the exam. A uniform guess over the real option counts is 0.318. It picks the first letter 47.9% of the time. The correct answer is in that slot 26.9% of the time.
5. At these prompt lengths the wait is decode. One forward pass is about 65–78 ms. A forced 32-token paragraph is about 2 seconds.
6. Training teaches the percentage list with letter cross-entropy. Typing the letter `A` would teach a pick, and the loss would never see the percentage.
7. Open sets are the bulk of the rows. A script owns the answer key. A larger model writes prose only for the gaps.
8. Shuffling about 30% of train rows moves the correct percentage with the word, so “always pick the first slot” stops being a shortcut.
9. Train loss fell from 1.487 to a last-100-step mean of 0.246. That is the practice pile. It is a different measurement from the exam.
10. Temperature changes how sure the percentages look. The winning letter stays put. Yes/no wanted 1.65. Multiple choice wanted 1.30. Score has no calibration rows, so it stays at 1.
11. Calibration accuracy near 0.91 is the same kind of question as training. The exam, before any LoRA, was 0.376. The ceiling on that exam is about 73.5%, because the labels agree with themselves about that often.

## What the post can say now

The method, the stack, the Phase 0 exam, the latency, the data cut, the training curve, and the calibration temperatures.

## What the post waits to say

Whether the LoRA beat plain E4B on typed-decisions. Whether the first-slot habit survived a flipped option order. How the model scores on SNLI, which never entered training. Those three are Phase 4.

A sentence that is safe today: the calibration pile shows the dial works, and the exam is the next measurement.

## Suggested order

1. The job, with one example.
2. How the percentage is read, and why the slot is a letter.
3. The exam, the ceiling, and what plain E4B did.
4. Latency: prefill against decode.
5. The rows: open data, generated gaps, the three piles, the shuffle.
6. The training loop: Unsloth as the loader, PyTorch as the loss, the curve, the unmerged file.
7. Temperature, with the two fitted values.
8. Close on what is still unmeasured.

## 1. The job

Situation in, closed list of answers in, percentage per answer out.

```text
state: "I was charged twice. Please refund."
question: "Which team?"
options: billing, tech, sales
output: billing 0.91, tech 0.06, sales 0.03
```

Those three numbers are the product. A route can fire when billing is above 0.85, and a person can see the rest. A paragraph is extra text the caller has to parse.

This is the same shape as a small judge: yes/no, a label from a list, or a score across ordered levels. The list is closed. The model is not asked to invent a fourth team.

## 2. How the percentage is read

A token is one piece of text the model knows as one id. A logit is the raw score for a token at one position. Softmax turns a few logits into percentages that are positive and sum to 1.

The readout is one forward pass. Take the logits at the last real token. Keep the option slots. Softmax those. Decode never starts.

The slot has to be one token. Gemma splits `human_review` into three pieces and `harmful` into two. Those strings have no single softmax slot. Every question therefore binds options to `A`, `B`, `C`, in the order the row lists them, and the softmax is over the letter ids. The option id and its description stay in the prompt, so the model still sees the words.

Two prompts exist on purpose. Training and calibration print `A. billing`. The exam prints `A. id: text`. The exam comparison has to keep the exam prompt, or it becomes a different measurement.

Thinking is off. No temperature on the Phase 0 exam. Temperature is a later dial, fit on a different pile.

## 3. The exam, before any training

Exam: `LocalLLaMA/typed-decisions`, config `all`, split `test`. 400 cases, 2,000 questions. We never train on it and we never fit temperature on it.

The labels come from a teacher that agrees with itself about 73.5% of the time. That is the ceiling. A score a little above it would mean the model fit that teacher’s quirks.

Plain `google/gemma-4-E4B-it`, letter softmax, thinking off, `T = 1`:

| slice | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| all | 2000 | 0.376 | 0.366 | 0.905 |
| yes/no | 600 | 0.538 | 0.294 | 0.714 |
| choice | 600 | 0.288 | 0.384 | 0.971 |
| score | 800 | 0.320 | 0.407 | 0.999 |

A uniform guess over the real option counts is 0.318. There are 600 two-way questions, 1,100 four-way, and 300 five-way. The readout is 0.058 above that.

Mean confidence on the chosen letter is 0.742. The model states about 74% and is right 37.6% of the time. That gap is the ECE. Expected calibration error buckets questions by the top percentage and compares that percentage with how often the pick was right.

It picks the first letter 958 times out of 2,000 (47.9%). The correct answer is in that slot 538 times (26.9%). Picked against gold by slot: 958 vs 538, 454 vs 585, 289 vs 493, 263 vs 330, 36 vs 54. The habit lives in the weights. The readout reports it. A later training pass is what can move it. Dividing every logit by the same temperature leaves the winner in place.

Yes/no is the least weak slice, at 0.538. Choice is under the uniform line for that slice. Customer-service questions are the weakest workflow, at 0.288.

Two hundred yes/no items omit the criteria text. The gold label is still false or true, so the prompt uses a fixed pair of descriptions. Dropping them would have scored 1,800 questions.

## 4. Latency

Same GPU, an L40S. Batch size 1. Median of three runs. Thinking off. Same exam prompts.

| prompt | input tokens | one forward pass | also decode the letter | forced 32 new tokens |
|---|---|---|---|---|
| short yes/no | 176 | 65 ms | 131 ms | 1.94 s |
| mid choice | 375 | 65 ms | 134 ms | 1.95 s |
| longest in this exam | 685 | 78 ms | 144 ms | 2.10 s |

Prefill reads the prompt in one pass. Decode writes the answer one token at a time. Each generated token costs about 60 ms. From 176 tokens to 685, prefill moved from 65 ms to 78 ms. At these lengths the wait is the decode.

Decoding the letter still does the prefill, then writes two tokens, the letter and the stop, so it takes about twice as long. The judge stops after the forward pass. Softmax over a handful of letters is free next to that pass.

The LoRA, once it is on, is a small extra multiply inside that same pass. Temperature is a divide on a handful of logits after the pass.

## 5. The rows

Three piles, cut before any weight update:

| pile | rows | role |
|---|---|---|
| train | 40,820 | updates the LoRA |
| calibration | 4,000 | fit temperature only |
| held out | 2,000 | SNLI only, scored later |
| exam | 2,000 | typed-decisions test, never copied into these files |

Train by source: MultiNLI 20,947, Banking77 9,101, BoolQ 8,586, refund rule 1,457, passage-answers-question 729. 12,246 train rows, exactly 30%, had their options shuffled, and the 1.0 moved with the correct word. Calibration was not shuffled.

Why these sets:

- BoolQ is already yes/no.
- MultiNLI is entail, neutral, or contradict.
- Banking77 is a support intent. It has 77 labels and no option list, so each row shows the true intent plus 19 others, in a random order. If the truth were always first, the model could learn slot 0 again.
- Refund routing and “does this passage answer the question?” are absent from those three. A script picked the label first. `google/gemma-4-12B-it` wrote the email or the note. The writer never filled `target`.

SNLI uses the same three labels as MultiNLI and is stored only in the held-out file, so a later score can ask whether the model learned “entailment” or only the MultiNLI wording.

A row looks like this:

```json
{
  "state": "I was charged twice. Please refund.",
  "question": "Which team?",
  "options": ["billing", "tech", "sales"],
  "target": [1.0, 0.0, 0.0]
}
```

`target` is the answer key, in the same order as `options`. When the list is shuffled, the 1.0 moves with the word.

The loss on one row, for a single correct option, is `-log(percentage on the correct option)`. Correct and sure is a tiny loss. Wrong and sure is a large loss. When the key is a spread such as `[0.7, 0.2, 0.1]`, the loss is the weighted sum of `-log(p)` across the options. That sum is cross-entropy. KL divergence moves the weights the same way, because it differs by a piece that depends only on the answer key.

## 6. Training

Checkpoint: `google/gemma-4-E4B-it`, loaded as `unsloth/gemma-4-E4B-it`. The `-it` means instruction-tuned. The prompt is an instruction, and the chat template matches that second training pass.

Gemma 4 E4B can also take an image or a sound. These rows are text. The LoRA is attached to the language attention and MLP only. Vision and audio stay off.

Unsloth is the loader. It attaches a LoRA of rank 16 and alpha 16. The loss is a short PyTorch loop. `SFTTrainer` is not in the loop, because supervised fine-tuning checks whether the next word was the right word. That teaches the model to type `A`. This loop teaches the percentage list.

One epoch. 40,820 rows. 5,103 steps. Batch 8. Max length 2,048. AdamW, learning rate `2e-4`, 100 warmup steps, cosine down to a tenth of that rate, gradient clip 1.0, seed 0. Right padding. The loss is read at the last real token.

588 tensors were trainable: 42 layers, and in each layer the query, key, value, output, gate, up, and down projections, each with a LoRA A and a LoRA B. `42 × 7 × 2 = 588`. Every name is under `language_model`.

The loss at step 1 was 1.487. The mean of the logged points at steps 1, 26, 51, and 76 was 0.975. The mean of the last 100 steps was 0.246.

That number is only over the option letters. A confused 2-way question sits near `log(2) ≈ 0.69`. A confused 20-way question sits near `log(20) ≈ 3.0`. Unsloth’s note that this model often shows a loss of 13–15 is a softmax over the whole vocabulary. It is a different number.

The saved file is a PEFT LoRA, about 140 MB. It stays a separate file. Merging it into Gemma would change the copy you might also use for writing. GitHub rejects a file over 100 MB, so the weights live on a Modal volume, `phase2-lora`, at `adapter/`.

Two practical notes worth a sidebar:

- The PyPI torchvision wheel does not load next to the cu128 torch build. The image installs Unsloth, then reinstalls `torchvision==0.26.0` and `torchaudio==2.11.0` from the cu128 index.
- A Modal client left connected for about 21 minutes had the GPU input cancelled. The epoch was finished as separate calls of about 10 minutes. Each call saved the LoRA and the next call loaded it. Each call also started a fresh Adam state, so the running average of past gradients reset at the boundary. The learning rate still followed the global step. The loss still fell across the epoch.

Machine: Modal L40S, 16-bit, one container per call, scaled down when the call ends.

## 7. Temperature

After training, the LoRA is frozen. One positive number `T` rescales the logits:

```text
percentages = softmax(logits / T)
```

`T = 1` leaves the percentages alone. `T > 1` makes the top answer look less sure. `T < 1` makes it look more sure. The winning option stays the same, so accuracy stays the same. ECE can move, because ECE is about the stated percentage.

The search is a grid from 0.50 to 3.00 in steps of 0.05. The chosen `T` is the one with the lowest mean `-log(percentage on the correct option)`. The fit reads the 4,000 calibration rows only. It does not read the train pile, SNLI, or the exam.

| type | rows | T | accuracy | ECE at 1 | ECE at T | Brier at 1 | Brier at T |
|---|---|---|---|---|---|---|---|
| yes/no | 912 | 1.65 | 0.909 | 0.055 | 0.015 | 0.143 | 0.133 |
| multiple choice | 3,088 | 1.30 | 0.923 | 0.032 | 0.009 | 0.119 | 0.116 |

Both values are above 1, so on this pile the model was a bit too sure. Accuracy at the chosen `T` matches accuracy at 1. There is no score row in the calibration pile, so score has no temperature and stays at 1 until a score calibration set exists.

Say this next to the 0.91: these rows are BoolQ, MultiNLI, Banking77, refund rules, and passage yes/no. They never updated the weights, and they are the same kinds of questions as the train pile. Phase 0 on the public exam was 0.376. The 0.91 is the dial check. The exam is the next measurement.

Brier is the third number. It gets worse when the pick is wrong and when the percentage was too extreme. Reporting it next to accuracy and ECE keeps a model from looking good on only one of them.

## 8. Close

The plain model is a weak judge with a first-slot habit, and it is over-sure. One forward pass is enough to read the percentages, and it is much shorter than writing them out. The LoRA was trained to move those percentages, and a single temperature per question type was fit afterward. The file is still a separate adapter.

The remaining measurement is the same 2,000 exam questions, with `softmax(logits / T)`, plus a flipped option order, plus SNLI. Keep the LoRA if accuracy is up and ECE is down against the Phase 0 table, on data it was not trained on.

## Numbers to keep exact

Rounded in the prose above. Exact floats:

- Phase 0: accuracy 0.376, ECE 0.3658742327145903, Brier 0.9052791643922966. Mean confidence 0.7418742327145903. First slot picked 958 / 2000, gold in that slot 538 / 2000.
- Latency file: `results/phase0/latency.json`.
- Phase 2: step 1 loss 1.486908, early mean of steps 1, 26, 51, 76 = 0.97460325, last-100 mean 0.24618104. The saved adapter directory is 179,090,427 bytes. The weights file inside it is about 140 MB.
- Latency medians: short 64.5 / 130.7 / 1944.7 ms, mid 65.4 / 134.0 / 1954.8 ms, longest 77.6 / 143.5 / 2101.6 ms. Those three clocks are one forward pass, decoding the letter, and a forced 32 new tokens.
- Phase 3: yes/no `T = 1.65`, NLL 0.26683733964031514 → 0.22779646590519353, ECE 0.054589416184109084 → 0.014922792529572435. Choice `T = 1.3`, NLL 0.2304244724952239 → 0.21622760975523075, ECE 0.03188975182841046 → 0.009062708872826912.

Files: `results/phase0/baseline.json`, `results/phase2/train.json`, `results/phase3/temperature.json`. The reasons for each design choice are in `LEARNING.md`.
