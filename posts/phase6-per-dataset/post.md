# One dataset, one LoRA, one before-count

8 October 2026. Model: `unsloth/gemma-4-E4B-it`, the Unsloth copy of `google/gemma-4-E4B-it`. Hardware for the scores: one Modal L40S. No adapter is loaded in the numbers below.

This post records the Phase 6 measurement that replaces the mixed-pile exam. It covers the protocol, the exact prompt and readout, the four splits, and the plain-model scores. All four plain tests are in. SNLI’s plain transfer score is in. None of the four LoRAs has been trained yet. A keep decision is not in this post. Each finished step is logged below with what was done, what the numbers were, and what is worth keeping for a later post.

## What the system returns

A situation and a closed list of answers go in. One forward pass runs. The model does not write a sentence. The next-token distribution is restricted to the option letters, and those logits become a percentage per answer:

```
softmax(letter_logits / T)
```

`T` is a temperature. At the plain-model scores in this post, `T = 1`. Temperature moves the percentages. It does not change which letter wins, because dividing every logit by the same positive number leaves the argmax where it was.

The letter is a slot, not the skill. `human_review` is more than one Gemma token, so a softmax over the vocabulary has no single cell for that string. Each answer is bound to `A`, `B`, `C`, … in the order printed in the prompt. The words of the answer stay in the prompt, to the left of the position we read.

The loss that will train each LoRA is letter cross-entropy: the log-softmax is taken over the option letters only, and the rest of the vocabulary is ignored. BoolQ, MultiNLI, and Banking77 use a one-hot target on the dataset’s own label. typed-decisions stores a probability vector from the teacher, so that vector is the training target. The published correct count on typed-decisions is still agreement with the stored label, which is the mode of that vector on every row we checked (`stored_label_not_mode: 0`). Ninety-six rows have a tie at that mode. The tie does not move the stored label.

## The input the model actually sees

Every dataset, including typed-decisions, uses one system line and one user layout. The system line is:

```
You are a judge. The next token must be one of the option letters. Do not explain.
```

The user message is:

```
State:
{situation}

Question: {question}
Options:
A. {option text}
B. {option text}

Reply with the single letter of the best option.
```

`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` turns that pair of messages into the token sequence. Training and scoring call the same function, `phase2.prompts.render_prompt`. The prompt hash frozen into every manifest is `06bf695261cfc389a19f9b5b0476e0c9cb1d6384c98869068a6a49ea1d22b79b`.

On a typed-decisions row the first option line is the criterion sentence, for example:

```
A. Let the agent proceed without interruption.
```

The option id `continue` is stored beside the row so the flip test can follow the answer when the letters move. It is not printed as `A. continue: …`. That `id: text` line was the old exam prompt. It is a different conditioning context, so the old 1,289 / 2,000 is not the before-count for this protocol.

Banking77 prints the intent with underscores turned into spaces, twenty names per row: the true intent plus nineteen others. The twenty are drawn once, when the split file is written, and then both the plain score and the later LoRA score read that file. MultiNLI and SNLI print `entailment`, `neutral`, and `contradiction`. BoolQ prints `no` and `yes`, in that order, until a train row is one of the rows whose options are shuffled.

## Where the percentage is read

Gemma’s tokenizer pads on the left. The old exam used `attention_mask.sum() - 1`. On a left-padded row that index is the first real token, and on a short row it can land on the padding. The published pair 752 / 2,000 to 941 / 2,000 is both models read at that early index. At the last real token, under the old exam prompt, plain Gemma is 1,289 / 2,000 and the first LoRA is 1,208 / 2,000.

Phase 6 builds the mask itself and pads on the right. Under right padding, `sum(mask) - 1` and the last index whose mask bit is 1 are the same position. The scorer checks that equality on every row and aborts if a content token sits to the right of a pad. The three finished tests passed that check on the original order and on the reversed order:

| test | rows | asserts | rows truncated at 2048 |
|---|---:|---:|---:|
| BoolQ validation | 3,270 | 6,540 | 0 |
| Banking77 test | 3,076 | 6,152 | 0 |
| typed-decisions test | 2,000 | 4,000 | 0 |

The assert count is twice the test count because the position-bias pass reverses the options and scores the row again. No test prompt was longer than 2,048 tokens.

## Why the mixed pile could not answer this

The retired run trained one LoRA on BoolQ, MultiNLI, Banking77, and a small script-labeled set, then graded it on typed-decisions. Those are different label sets, different wording, and different option counts. After that grade, a miss is the sum of two effects: the update failed to learn the practice tasks, and the update failed to carry onto invoices, traces, and incidents. The two effects cannot be separated from one number.

The in-family scores had the other hole. Calibration was about 3,680 / 4,000. SNLI, held out of training, was 1,820 / 2,000. Plain Gemma was never scored on those files, so a high count after training has no paired count before training.

typed-decisions was forbidden as a training signal and used as the only public scoreboard. Its stored letter is the average of three samples from another model at temperature 0.7. A fresh sample from that teacher matches the stored letter about 1,470 / 2,000 times. That is the ceiling of agreeing with the file. The dataset’s own train split is about 6,000 questions. Leaving those questions out meant the run never estimated whether this model can learn that teacher.

One shared LoRA also hides a trade. BoolQ is 2-way. MultiNLI is 3-way. Banking77 is 20-way. typed-decisions mixes yes/no, choice, and ordered score. A gain on one of those can be paid for by a loss on another, and a single keep bit still passes.

## The comparison this run estimates

For each dataset D:

1. Freeze a test set from D. No test row is a training row or a temperature row.
2. Score plain Gemma on that test. Record the correct count, ECE, and Brier.
3. Train a new LoRA from the frozen base on D’s train rows only.
4. Score that LoRA on the same rows, the same prompt, and the same token index.
5. Fit one temperature on a calibration cut taken from D’s train side. Accuracy at that temperature must equal accuracy at `T = 1`.
6. Keep the LoRA for D when the correct count went up and ECE went down.

A keep on D says nothing about any other dataset. SNLI is a transfer check for the MultiNLI LoRA only. It does not enter that keep bit.

The adapters will be saved at `/lora/v2-boolq`, `/lora/v2-multinli`, `/lora/v2-banking77`, and `/lora/v2-typed-decisions`. They are not merged into Gemma. They are not started from `/lora/adapter` or `/lora/adapter-pass2`. The Phase 1 guard that rejects a typed-decisions source is still in `phase1/rows.py`. The new rows live in `phase6/`, which is allowed to name that source.

Training, once it runs, is rank 16, alpha 16, dropout 0, text and attention and MLP only, vision and audio off, 16-bit, one epoch, batch 8, max length 2048, AdamW, learning rate `2e-4`, warmup 100 steps, cosine down to `0.1` times the learning rate, gradient clip 1.0. Each Modal call trains for about eight minutes and then saves. The next call resumes that dataset’s checkpoint with a fresh AdamW. The learning rate follows the global step.

## How the splits were cut

Each file was rebuilt from the download. `data/phase1/train.jsonl` was not reused. Calibration is 2,000 rows taken at random from the train side, then written back in source order. Thirty percent of the remaining train rows have their options shuffled, and the target moves with the option id. Calibration and test are not shuffled. Train row order is shuffled so one epoch is not blocked by the original file order. Seed 0.

typed-decisions is cut by case. A case is five questions about one situation. All five stay on the same side of the calibration cut, so a situation used to fit `T` is not also a situation whose weights were updated. Every case contains yes/no (`noul`), choice, and score, so the calibration file contains all three. The test is the published 400-case, 2,000-question split.

| dataset | downloaded train side | after the cut | calibration | test |
|---|---:|---:|---:|---:|
| BoolQ | 9,427 train passages | 7,427 train | 2,000 | 3,270 validation |
| MultiNLI | 392,702 train pairs | 390,571 train | 2,000 | 9,815 matched validation |
| Banking77 | 9,993 train utterances | 7,987 train | 2,000 | 3,076 official test |
| typed-decisions | 1,200 cases, 6,000 questions | 4,000 questions | 2,000 questions | 2,000 questions |
| SNLI | — | — | — | 9,842 validation rows |

MultiNLI’s official train repeats 131 premise/hypothesis pairs. On 103 of those dropped copies the label disagreed with the copy that was kept. The later copy was dropped before the cut, so the same pair cannot sit in both train and calibration. The matched validation split shares no pair with the train side.

Banking77’s official test repeats 6 utterances that are also in the official train. Those 6 were removed from the train side. The test file is unchanged. Each remaining row holds the true intent plus 19 distractors drawn from the 77 names. The test draw uses seed 1. The train draw uses seed 0. Both draws are stored in the row.

SNLI validation has 10,000 examples. 158 have no label and were dropped. Those 9,842 rows are the transfer file. They are not a fifth LoRA.

The MultiNLI train file is 261,524,070 bytes. It is gitignored. Its sha256 is in `data/phase6/multinli/manifest.json`. `python -m phase6.build` rebuilds it.

typed-decisions calibration types: choice 602, yes/no 598, score 800. Test types: choice 600, yes/no 600, score 800. Train types: choice 1,198, yes/no 1,202, score 1,600. Shuffle fraction on every train split is 0.30. No calibration row and no test row is marked shuffled.

1,682 typed-decisions targets were rescaled so the teacher probabilities sum to 1. The raw sums were within `1e-3` of 1. The stored label was a mode on every row.

## Plain Gemma, before any LoRA

Same prompt, right padding, last real token, `T = 1`, base weights.

| dataset | correct | accuracy | ECE | Brier | flips when the options are reversed |
|---|---:|---:|---:|---:|---:|
| BoolQ validation | 2,824 / 3,270 | 0.864 | 0.117 | 0.248 | 86 / 3,270 |
| Banking77 test | 2,547 / 3,076 | 0.828 | 0.120 | 0.285 | 400 / 3,076 |
| MultiNLI matched validation | 7,393 / 9,815 | 0.753 | 0.183 | 0.419 | 1,051 / 9,815 |
| typed-decisions test | 1,219 / 2,000 | 0.610 | 0.303 | 0.661 | 305 / 2,000 |
| SNLI validation, transfer only | 6,140 / 9,842 | 0.624 | 0.349 | 0.715 | 727 / 9,842 |

The base model is already well above a random pick. These rows are the before-count. A later LoRA has to pass them.

A random letter on BoolQ is half the validation set, 1,635 / 3,270. Plain Gemma is at 2,824 / 3,270. A random letter on Banking77, twenty names on every row, is about 154 / 3,076. Plain Gemma is at 2,547 / 3,076. The intent name is written in the prompt, and one forward pass is enough to match the customer sentence to that name. A random letter on typed-decisions, using the real option counts, is about 635 / 2,000. Plain Gemma is at 1,219 / 2,000. The teacher repeats its own stored letter about 1,470 / 2,000 times, so 1,219 is most of the way from a guess to that ceiling and still about 250 questions short of it.

No LoRA produced these counts. The older 1,289 / 2,000 is the same plain weights on the same 2,000 questions with the old line `A. {id}: {text}`. The 70-question difference is the prompt. BoolQ and Banking77 set a high bar for training. MultiNLI sets it at 7,393 / 9,815. typed-decisions sets the bar at 1,219, with 1,470 as the limit of agreeing with the file. SNLI, 6,140 / 9,842, is the transfer before-count and is not a keep row.

The three columns are three different measurements.

The correct count is how often the chosen letter is the letter in the answer key. On BoolQ and Banking77 that key is the dataset’s own label. On typed-decisions it is the teacher’s stored letter.

ECE is how far the percentage on that chosen letter sits from the hit rate. The hit rate is the correct count as a fraction: BoolQ is right on about 86 of every 100 rows. Questions are grouped by the percentage the model stated, and each group is checked against how often that group was actually right. BoolQ’s ECE is 0.117, about a 12-point gap. Banking77 is 0.120. typed-decisions is 0.303, about a 30-point gap, and the stated percentage sits above the hit rate. The model is more sure on those rows than its results.

The flip count is how often the chosen answer changes when the same words are moved to different letters. It is a position report. BoolQ changes on 86 / 3,270 rows, so the letter is following `yes` and `no`. Banking77 changes on 400 / 3,076. typed-decisions changes on 305 / 2,000. The keep rule does not use this count.

ECE is the expected calibration error of the top-letter percentage against whether that letter was right, in ten equal-width bins. Brier is the mean squared error of the full percentage vector against a one-hot label. On typed-decisions that one-hot is the stored teacher label.

A uniform guess on BoolQ is 1,635 / 3,270. Plain Gemma is 2,824 / 3,270. Reversing `no` and `yes` changes the chosen word on 86 rows, 2.6%. The letter is following the word.

A uniform guess on Banking77, with twenty names on every row, is about 154 / 3,076. Plain Gemma is 2,547 / 3,076. The intent name is in the prompt, and the model is using it. Reversing the twenty names changes the chosen intent on 400 / 3,076 rows, 13.0%. That is the position-bias report for this test. It is not the keep bit.

typed-decisions by question type, still the plain model:

| type | correct | accuracy | ECE | Brier |
|---|---:|---:|---:|---:|
| choice | 360 / 600 | 0.600 | 0.353 | 0.730 |
| yes/no (`noul`) | 376 / 600 | 0.627 | 0.305 | 0.647 |
| score | 483 / 800 | 0.604 | 0.269 | 0.620 |
| all | 1,219 / 2,000 | 0.610 | 0.303 | 0.661 |

A uniform draw over the real option counts on this exam is about 635 / 2,000. The teacher agrees with the stored letter about 1,470 / 2,000 times. Plain Gemma, on the frozen `A. {option text}` prompt, sits between those two: 1,219 / 2,000.

The earlier last-token number, 1,289 / 2,000, used `A. {id}: {text}`. This protocol’s before-count is 1,219 / 2,000. The 70-question gap is a prompt change on the same questions and the same token index. It is not a training result. The LoRA, when it is trained, has to beat 1,219 / 2,000 on this prompt, and its ECE has to fall below 0.303, or that adapter is not kept.

The typed-decisions percentages are the over-sure ones. BoolQ is right on 86% of rows with ECE 0.117. Banking77 is right on 83% of rows with ECE 0.120. typed-decisions is right on 61% of rows with ECE 0.303 and Brier 0.661. Choice is the worst calibrated of the three types, ECE 0.353. The model’s top-letter percentage is farther from its hit rate on these traces, invoices, and incidents than it is on yes/no passages or on banking intent names.

Reversing typed-decisions options changes the chosen option id on 305 / 2,000 rows, 15.3%. The old flip report, 1,254 / 2,000, was the first LoRA, the old exam prompt, and the early index. These are different treatments. The 305 is the position report for plain Gemma under the new prompt.

## What a later keep is allowed to say

A kept Banking77 LoRA will mean the letter readout got more of this official test right, and the percentages moved closer to the hit rate, under this prompt and this index. It will not mean the adapter is a judge for a new bank’s private categories. Those categories need their own held-out tickets. The plain model already gets 2,547 / 3,076 by reading intent names that were written in the prompt. The LoRA has to beat that, not beat a 5% guess.

A kept typed-decisions LoRA will mean the model moved closer to the teacher on the 2,000 held-out questions. It will not mean the invoices are judged correctly past the teacher’s own agreement, about 1,470 / 2,000.

A miss on one dataset stays on that dataset.

## Step log: plain MultiNLI and plain SNLI

Finished 8 October 2026, 12:35 UTC. One Modal call, base weights, no LoRA. The call scored MultiNLI matched validation and, in the same process, the SNLI validation file. Right padding. Last content token. `T = 1`. Prompt hash `06bf695261cfc389a19f9b5b0476e0c9cb1d6384c98869068a6a49ea1d22b79b`. Truncated rows: 0. Padding asserts: 19,630 on MultiNLI and 19,684 on SNLI, which is two passes each, original order and reversed order.

### What the run did

`modal run phase6/modal_score.py --dataset multinli --weights plain` loaded `unsloth/gemma-4-E4B-it`, built each batch with an explicit right pad, and aborted if `mask.sum() - 1` was not the last 1 in the mask. It did not. The letters were read at that index. Files: `results/phase6/multinli/plain.json` and `results/phase6/snli/plain.json`.

### What it found

MultiNLI matched validation: 7,393 / 9,815 correct. Accuracy 0.753. ECE 0.183. Brier 0.419. Reversing entailment, neutral, and contradiction changed the chosen label on 1,051 / 9,815 rows, 10.7%.

SNLI validation, same three words, plain weights, full file after dropping 158 unlabeled rows: 6,140 / 9,842 correct. Accuracy 0.624. ECE 0.349. Brier 0.715. Flips: 727 / 9,842, 7.4%.

A uniform draw over three labels is about 3,272 / 9,815 on MultiNLI and about 3,281 / 9,842 on SNLI. Both scores are above that. MultiNLI is about 2.3 times a uniform draw. SNLI is about 1.9 times.

### What it means

The three label words are not the skill. MultiNLI and SNLI print the same options, `entailment`, `neutral`, and `contradiction`, under the same system line. Plain Gemma is 13 points higher on MultiNLI than on SNLI, and SNLI’s ECE is almost twice MultiNLI’s. The stated percentage on SNLI sits much farther from the hit rate. Same letters, different sentences, different calibration.

This SNLI file is the full validation split, 9,842 labeled rows. The retired run’s SNLI number, 1,820 / 2,000, was a 2,000-row sample scored after the mixed LoRA, with choice temperature 1.30. Those two fractions do not share a denominator. The new before-count for transfer is 6,140 / 9,842 at `T = 1` on the plain model. The MultiNLI LoRA, when it exists, is the model that gets compared with that count. The comparison does not enter the keep bit.

MultiNLI’s before-count for the keep bit is 7,393 / 9,815, ECE 0.183. A later LoRA is kept on this dataset only if the correct count rises above 7,393 and ECE falls below 0.183 on this same test.

The flip rates, 10.7% and 7.4%, say the chosen word usually survives a reversal of the three lines. The position report is quieter here than on Banking77 (13.0%) and typed-decisions (15.3%). BoolQ remains the quietest, 2.6%, because `yes` and `no` are hard to confuse with a slot.

### Blog lines from this step

- Same three words, two datasets: 7,393 / 9,815 on MultiNLI, 6,140 / 9,842 on SNLI. The label list is not the task.
- The plain model is already a 75% entailment judge on the official matched set. Training has to beat that, not beat one third.
- SNLI is the over-sure plain score: right on 62% of rows, ECE 0.349. A later LoRA that reaches a high count with a low ECE is a change in both accuracy and calibration. A high count alone is the mistake the old 1,820 / 2,000 could not catch, because the plain count was never written down.
- Do not put 1,820 / 2,000 next to 6,140 / 9,842 as if they were the same exam.

## What is still ahead

Four LoRAs, trained from the base, one at a time, in this order: BoolQ, Banking77, typed-decisions, MultiNLI. MultiNLI is last because its train file is 390,571 rows, about 48,822 steps at batch 8. Each LoRA is then scored with the same code path as the plain run. Temperature is fit on that dataset’s 2,000 calibration rows, grid 0.50 to 3.00 in steps of 0.05, lowest mean letter negative log likelihood. typed-decisions calibration has yes/no, choice, and score, so none of its test types are forced to stay at `T = 1` for lack of rows. Accuracy at the chosen `T` must match accuracy at `T = 1`. ECE is recomputed at the chosen `T`. The keep table is four rows. SNLI is scored again with the MultiNLI LoRA and stays out of the bit.

The procedure is `PROTOCOL.md`. The implementation is `phase6/`.
