# Blog notes

Source notes for the post. The post is not written here. Phase 4 has been run. The comparison with Phase 0 is the shared index, 0.376 to 0.471.

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
11. Calibration accuracy near 0.91 is the same kind of question as training. The exam, before any LoRA, was 0.376. With the LoRA, on that same index, it is 0.471, and ECE fell from 0.366 to 0.247. The ceiling on that exam is about 73.5%, because the labels agree with themselves about that often. The last real token on the LoRA pass scores 0.604, and Phase 0 has not been read at that token.

## Key insights

These are the lines worth building the post around. Each one is already measured.

**The product is the percentage list.** A situation and a closed set of answers go in. A percentage per answer comes out. Code can threshold it. The model stops before it writes a paragraph.

**The slot has to be one token.** `human_review` is three Gemma tokens and `harmful` is two. A softmax has one slot per answer, so every question is bound to `A`, `B`, `C` in dataset order. The words stay in the prompt. Training prints `A. billing`. The exam prints `A. continue: Let the agent proceed`. The exam comparison keeps the exam wording.

**Plain E4B is a weak, over-sure judge with a first-slot habit.** On 2,000 exam questions, accuracy is 0.376. A uniform guess over the real option counts is 0.318. Mean confidence on the top letter is 0.742, so it talks like it is right about three times in four and is right a bit over one time in three. It picks the first letter 958 times (47.9%). The correct answer is there 538 times (26.9%). The habit is in the weights. Reading the logits reports it.

**The wait is the decode.** On an L40S, one forward pass is 65–78 ms at these prompt lengths. Writing the letter is about 130–145 ms. A forced 32-token paragraph is about 2 seconds, roughly 60 ms per generated token. The judge never starts that second clock.

**Open labels teach the skill. A script owns the answer key.** BoolQ, MultiNLI, and Banking77 are the bulk: 40,820 train rows. About 30% of them (12,246) had the option order shuffled, and the correct percentage moved with the word. Refund emails and passage notes were written by Gemma 4 12B. The script picked `target` before the writer ran. The exam, typed-decisions, was never copied into training. Its labels agree with themselves about 73.5% of the time. That is the ceiling.

**Letter cross-entropy is a small number on purpose.** The loss is only over the option letters. Step 1 was 1.487. The last 100 steps averaged 0.246. A confused yes/no sits near `log(2) ≈ 0.69`. A confused 20-way question sits near `log(20) ≈ 3.0`. A full-vocabulary loss of 13–15, which shows up in Unsloth’s notes for this model, is a different softmax. Unsloth loaded the model and attached the rank-16 LoRA. The PyTorch loop applied the loss. The saved file is a separate 140 MB adapter. A falling train loss means the practice pile moved. The exam is the other file.

**Temperature changes the stated percentage. The winner stays put.** `softmax(logits / T)`. The fit used 4,000 calibration rows the weights never saw, and none of the exam. Yes/no landed at 1.65. Multiple choice landed at 1.30. Both are above 1, so on that pile the model was a bit too sure. Score has no calibration rows, so score stays at 1. Calibration accuracy near 0.91 is BoolQ, MultiNLI, Banking77, and the small generated sets. Same family as training.

**On the shared exam index, the adapter is worth keeping.** Accuracy 0.376 to 0.471. ECE 0.366 to 0.247. Brier 0.905 to 0.737. Multiple choice did the work, 0.288 to 0.518. Yes/no was already the strong slice and stayed at 0.540. Score, absent from training, moved from 0.320 to 0.383. 0.471 is under the 73.5% ceiling, so this is not a clone of the exam’s teacher.

**Left padding splits the headline in two.** Shorter questions in a batch have blank tokens on the left. `mask.sum() - 1` lands on the first content token. The last content token is the last index where the mask is 1. Those two positions choose different option ids on 824 of 2,000 questions for the LoRA, and on 1,228 of 2,000 for the plain model. Publish 752 to 941 as the training comparison, because both used `sum - 1`. At the last content token the plain model scores 1,289 of 2,000 and the LoRA scores 1,208 of 2,000. Those two forwards used different stacks (transformers 5.18.0 versus Unsloth), so 1,289 versus 1,208 is not a pure training delta.

**The first-slot habit shrank and did not leave.** Reversing the option list changed the chosen option id on 1,254 of 2,000 questions (62.7%). Slot A still gets 762 picks against 538 gold labels. Phase 0 put 958 picks there. Temperature cannot fix this. Dividing every logit by the same number leaves the winner in place.

**SNLI is transfer of a label set the model already trained on.** 2,000 rows, never in the train file, same three labels as MultiNLI. Accuracy 0.910, ECE 0.013, on the training prompt with right padding. That sentence belongs next to the exam number, with a clear label. It does not decide whether the adapter stays.

## Number to report

The readout index was wrong. Gemma left-pads. Picture a short row as `PAD PAD t1 t2 t3`. `attention_mask.sum() - 1` lands on `t1`. The last content token is `t3`. Those indexes pick different letters on 824 of 2,000 LoRA questions and on 1,228 of 2,000 plain-model questions.

Report the training delta on one index. Both Phase 0 and Phase 4 stored `sum - 1`. The plain re-score reproduced 752 exactly (`shortcut_matches_phase0` is true):

| | correct |
|---|---|
| plain model, `sum - 1` | 752 / 2,000 |
| LoRA, `sum - 1` | 941 / 2,000 |

The same plain forward, read at the last content token, is 1,289 / 2,000. Per type: choice 379 / 600, yes/no 433 / 600, score 477 / 800. Workflows: agent trace 258 / 500, customer service 375 / 500, invoice 323 / 500, security 333 / 500. ECE 0.2624814863356488, Brier 0.5983526449027284. No LoRA, temperature 1, the Phase 0 prompt, transformers 5.18.0.

The LoRA at the last content token, from the Phase 4 Unsloth forward, is 1,208 / 2,000. At the right token the untouched model is ahead by 81 questions. At the shared wrong index the LoRA is ahead by 189 questions (941 − 752). Do not publish 752 to 1,289, or 752 to 1,208, as the training effect. Those mix the index fix with training, and the last-token pair also mixes two stacks.

## What I would do next

The plain-model read at the last content token is in: 1,289 / 2,000, file `results/phase4/plain_last.json`. The shared-index move from 752 / 2,000 to 941 / 2,000 stays the training comparison. The last-token pair is plain 1,289 versus LoRA 1,208, with the stack caveat above.

A second train is the remaining gap. The first train file has no ordered-score rows. Slot 0 is still over-picked. The second pass uses 2,186 score rows (refund severity from a phrase in the email, passage counts from listed facts) plus 4,000 fully reshuffled choice rows from MultiNLI and Banking77. BoolQ and the exam stay out. Learning rate 2e-5, one pass, saved as a new adapter file so the first adapter is not overwritten. Score that adapter on the exam before claiming a new count.

Another epoch on the same 40,820 rows is not. Letter loss already fell from 1.487 to about 0.25. More BoolQ yes/no is not either: exam yes/no went from 323 / 600 to 324 / 600 on the shared index. Leave both adapters unmerged.

## What the post can say now

The method, the stack, the Phase 0 exam, the latency, the data cut, the training curve, the calibration temperatures, and the Phase 4 exam.

The exam comparison uses the same index as Phase 0. On that index the LoRA is accuracy 0.471, ECE 0.247, Brier 0.737, against Phase 0 at 0.376, 0.366, and 0.905. The gate keeps the LoRA. Choice carried the gain (0.288 to 0.518). Yes/no stayed near 0.54. Score, absent from training, moved from 0.320 to 0.383.

Say the left-padding caveat next to those numbers. The tokenizer left-pads, and Phase 0 reads `mask.sum() - 1`. On the LoRA pass that position is the first content token for 824 of 2,000 questions. The last content token on the same pass scores 1,208 / 2,000. A plain re-score reproduces 752 at `sum - 1` and scores 1,289 / 2,000 at the last content token. 1,208 is not the comparison with 752.

The reversed option list changed the chosen id on 1,254 of 2,000 questions. Slot 0 is still over-picked: 762 picks, 538 gold labels. Phase 0 picked slot 0 on 958 questions.

SNLI, held out of training, scores accuracy 0.910 and ECE 0.013. Same three labels as MultiNLI. That is not the typed-decisions exam.

## What the post still leaves open

A Phase 0 re-score at the last real token. Until that exists, the published comparison is the shared index: 0.376 to 0.471.

## Suggested order

1. The job, with one example.
2. How the percentage is read, and why the slot is a letter.
3. The exam, the ceiling, and what plain E4B did.
4. Latency: prefill against decode.
5. The rows: open data, generated gaps, the three piles, the shuffle.
6. The training loop: Unsloth as the loader, PyTorch as the loss, the curve, the unmerged file.
7. Temperature, with the two fitted values.
8. The exam, the flip, and SNLI.
9. What is still open: a Phase 0 read at the last real token.

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

Phase 0 and the published Phase 4 comparison did not quite do that. The Gemma tokenizer pads on the left, and both runs read `mask.sum() - 1`. On a left-padded row the real text is flush right, so that index misses the last real token except on the longest row in the batch. The notes below keep both numbers.

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

Say this next to the 0.91: these rows are BoolQ, MultiNLI, Banking77, refund rules, and passage yes/no. They never updated the weights, and they are the same kinds of questions as the train pile. The typed-decisions exam, on the shared index, moved from 0.376 to 0.471. The 0.91 is the dial check on the calibration pile.

Brier is the third number. It gets worse when the pick is wrong and when the percentage was too extreme. Reporting it next to accuracy and ECE keeps a model from looking good on only one of them.

## 8. The exam

Same 2,000 questions as Phase 0. Same letter prompt. `softmax(logits / T)` with yes/no at 1.65, multiple choice at 1.30, and score at 1. The adapter file was not rewritten.

| | accuracy | ECE | Brier |
|---|---|---|---|
| plain E4B | 0.376 | 0.366 | 0.905 |
| LoRA, same index | 0.471 | 0.247 | 0.737 |

Accuracy went up and ECE went down. 0.471 is under the 73.5% ceiling. The gain is concentrated in multiple choice, 0.288 to 0.518. Yes/no was already 0.538 and landed at 0.540. Score was not in the training pile and moved from 0.320 to 0.383.

The tokenizer left-pads. Both Phase 0 and this run read `mask.sum() - 1`. For a shorter row in the batch that index is the first content token. The two positions pick different option ids on 824 questions for the LoRA. The last content token on this LoRA pass scores 1,208 / 2,000 (accuracy 0.604, ECE 0.162, Brier 0.550). A later plain-model forward, same exam prompts, no adapter, temperature 1, reproduces 752 at `sum - 1` and scores 1,289 / 2,000 at the last content token. Publish 752 to 941 as the training comparison. Publish 1,289 and 1,208 as the last-token pair, and say the stacks differ.

Reversing the options changed the chosen id 62.7% of the time (1,254 / 2,000). Slot 0 got 762 picks against 538 gold labels. The plain model put 958 picks there. The habit is smaller. It is still in the weights.

SNLI was left out of training. Accuracy 0.910, ECE 0.013, on the training prompt with right padding. The labels are entailment, neutral, and contradiction, which MultiNLI already taught. The number says that wording transferred. It does not say the typed-decisions exam transferred.

## 9. Close

The plain model at the old index is a weak judge with a first-slot habit, and it is over-sure. Read at the last content token, the same plain model gets 1,289 of 2,000. One forward pass is enough to read the percentages, and it is much shorter than writing them out. The LoRA moved the exam from 752 to 941 on the shared index, and the stated percentages got closer to the hit rate. The adapter is still a separate file. The first-slot habit shrank and did not disappear. At the last content token the LoRA is 1,208 of 2,000, which is below the plain model’s 1,289 on a different stack.

## Numbers to keep exact

Rounded in the prose above. Exact floats:

- Phase 0: accuracy 0.376, ECE 0.3658742327145903, Brier 0.9052791643922966. Mean confidence 0.7418742327145903. First slot picked 958 / 2000, gold in that slot 538 / 2000.
- Latency file: `results/phase0/latency.json`.
- Phase 2: step 1 loss 1.486908, early mean of steps 1, 26, 51, 76 = 0.97460325, last-100 mean 0.24618104. The saved adapter directory is 179,090,427 bytes. The weights file inside it is about 140 MB.
- Latency medians: short 64.5 / 130.7 / 1944.7 ms, mid 65.4 / 134.0 / 1954.8 ms, longest 77.6 / 143.5 / 2101.6 ms. Those three clocks are one forward pass, decoding the letter, and a forced 32 new tokens.
- Phase 3: yes/no `T = 1.65`, NLL 0.26683733964031514 → 0.22779646590519353, ECE 0.054589416184109084 → 0.014922792529572435. Choice `T = 1.3`, NLL 0.2304244724952239 → 0.21622760975523075, ECE 0.03188975182841046 → 0.009062708872826912.
- Phase 4, same index as Phase 0: accuracy 0.4705 (941 / 2000), ECE 0.24664896169448602, Brier 0.7370409524588989. Last content token on that pass: accuracy 0.604 (1208 / 2000), ECE 0.16193150770165962, Brier 0.5501747421983022. Index disagreement: 824 / 2000. Flip changes: 1254 / 2000. SNLI: accuracy 0.91, ECE 0.012526768167657777, Brier 0.13375256693900067.
- Plain model re-score, `results/phase4/plain_last.json`: `sum - 1` is 752 / 2000 and matches Phase 0. Last content token is 1289 / 2000, ECE 0.2624814863356488, Brier 0.5983526449027284. Per type at the last content token: choice 379 / 600, noul 433 / 600, score 477 / 800. Index disagreement: 1228 / 2000. Padding side left. No LoRA. Temperature 1.

Files: `results/phase0/baseline.json`, `results/phase2/train.json`, `results/phase3/temperature.json`. The reasons for each design choice are in `LEARNING.md`.
