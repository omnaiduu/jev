# Per-dataset training protocol

Status: active plan. The mixed-pile protocol in `PLAN.md` is retired. Do not start a GPU run from this file until asked. The historical adapters stay on disk and are not the starting weights for this run.

Goal: for each dataset, measure whether a LoRA on frozen Gemma 4 E4B raises accuracy and lowers ECE on rows from that same dataset. One dataset, one adapter, one before-count, one after-count.

## Why the old method is flawed

The old run asked one question in the writeup and measured a different one in the code.

The writeup asked: did the add-on become a better judge than plain Gemma? The procedure trained on BoolQ, MultiNLI, Banking77, and a small script-labeled set, then graded on typed-decisions. Those are different label sets, different wording, and different option counts. A miss on that grade is the sum of two effects: failure to learn the practice tasks, and failure to carry that learning onto invoices, incidents, and agent traces. After the run, the two effects cannot be separated. The 1,208 of 2,000 at the last real token does not say whether Banking77 got better. The 1,820 of 2,000 on SNLI does not say whether typed-decisions can be learned.

A fine-tune has a defined comparison. Same rows, same prompt string, same token index, weights before, weights after. The old exam changed the rows and the prompt. The in-family scores changed the weights and never recorded the before. Calibration was about 3,680 of 4,000 on more BoolQ, MultiNLI, Banking77, and the same rules. SNLI was 1,820 of 2,000. Plain Gemma was never scored on those files. A high count after training, with no paired count before training, does not show that training raised it.

typed-decisions was used as the only public scoreboard and was forbidden as a training signal. Its stored letter is the average of three samples from another model, temperature 0.7. A fresh sample from that teacher matches the stored letter about 1,470 of 2,000 times. That is the ceiling of “agree with this file.” Training on Wikipedia yes/no and banking intents, then scoring against that teacher, estimates neither “can this model learn the teacher” nor “can this model judge an invoice.” The 6,000 train questions of that file were the data that would have estimated the first of those. Leaving them out was the design, and it is the mistake. Holding the 2,000 test questions out of the weight update is what keeps the test honest. Holding the train questions out as well throws away the estimate.

One LoRA received every gradient. BoolQ is 2-way. MultiNLI is 3-way. Banking77 was shown as 1 correct intent plus 19 others. typed-decisions mixes yes/no, choice, and ordered score. The keep rule was a single bit on a foreign exam: accuracy up and ECE down. A gain on one label set can be paid for by a loss on another, and the bit still passes. The second pass made that concrete. It added score homework and a full shuffle, and the shared-index exam moved from 941 of 2,000 to 899 of 2,000.

The loss and the published exam index were different positions. Training padded blanks on the right, so `mask.sum() - 1` was the last real token, and letter cross-entropy taught that token. The exam tokenizer pads on the left. The same formula then counts from the left edge of a row whose text has been pushed right. On a short row it lands early, sometimes on a blank. The published pair, 752 to 941 of 2,000, is both models read at that early index. At the last real token the plain model is 1,289 of 2,000 and the first LoRA is 1,208 of 2,000. The gate kept an adapter for a comparison the loss did not train.

The prompt string also changed between train and exam. Training printed `A. {option}`. The exam printed `A. id: text`. The letter logit is conditioned on the tokens to its left. Two strings are two treatments. A delta across them is not a training effect.

Temperature cannot repair this. `softmax(logits / T)` changes the percentage and leaves the winning letter in place. T was fit on the practice family. Score on the exam had no calibration rows, so it stayed at 1. ECE on typed-decisions is a confidence report about a letter that was chosen under a different prompt and, in the published file, at a different token.

## What is retired

- One mixed train file of 40,820 rows graded on typed-decisions.
- A second pass that updates that same adapter and is graded on typed-decisions again.
- Publishing 752 to 941 as the training result.
- Starting the next adapter from `/lora/adapter` or `/lora/adapter-pass2`.
- Merging a LoRA into Gemma.
- Training on a test split, fitting T on a test split, or fitting T on typed-decisions test questions.

The old result files stay. They document the retired comparison. They are not the baseline for this protocol.

## The comparison this protocol estimates

For dataset D:

1. Freeze a test set T drawn from D. No row in T is ever a training row or a temperature row.
2. Score plain Gemma on T. Record correct count, ECE, and Brier.
3. Train a new LoRA from the frozen base on D’s train rows only.
4. Score that LoRA on the same T, same prompt, same token index.
5. Fit temperature on a calibration cut taken from D’s train side only. Accuracy at that T must equal accuracy at T = 1. Report ECE at the fitted T.
6. Keep that LoRA for D when the correct count went up and ECE went down. A keep on D says nothing about any other dataset.

typed-decisions uses the same steps. Its correct count means agreement with the teacher’s stored letter. Write the 1,470 of 2,000 ceiling next to every typed-decisions count. A higher count means closer to that teacher.

## Datasets

Each run rebuilds rows from the original dataset. Do not reuse `data/phase1/train.jsonl` as if it were a finished split. That file already mixed sources and already spent the public train rows.

| dataset | train side | test | what the label is |
|---|---|---|---|
| BoolQ | official train, then a calibration cut taken from it | official validation | human yes/no |
| MultiNLI | official train, then a calibration cut taken from it | official matched validation | entailment, neutral, contradiction |
| Banking77 | official train, then a calibration cut taken from it | official test | one of 77 intents |
| typed-decisions | the dataset’s train split, about 6,000 questions, then a calibration cut taken from it | the 2,000-question test split | teacher letter, three samples at temperature 0.7 |

Confirm every row count from the downloaded split and write it into the manifest. SNLI stays a transfer check for the MultiNLI adapter only. It does not enter that adapter’s keep bit. The script-labeled refund and passage rows are out. typed-decisions already contains ordered-score questions; those are the score rows for the typed-decisions run.

Banking77 has no natural option order. Each row is the true intent plus 19 other intents. Draw the 19 once, while the split is being written, and save the order in the row. The plain score and the LoRA score must read that same order.

Shuffle 30 percent of each train split. The correct percentage moves with the words. Do not shuffle calibration or test. After the test score, reverse the test options, rebind A, B, C, and count how often the chosen option id changes. That flip count is a position-bias report. It is not the keep bit.

## Model and loss

- Base: `google/gemma-4-E4B-it`, Unsloth id `unsloth/gemma-4-E4B-it`.
- Text, attention, and MLP only. Vision off. Audio off.
- 16-bit load. Rank 16, alpha 16, dropout 0, bias none.
- Unsloth loads the model and attaches the LoRA. A short PyTorch loop applies letter cross-entropy. Do not use SFTTrainer.
- One epoch, batch 8, max length 2048, AdamW, learning rate 2e-4, warmup 100 steps, cosine down to 0.1 times the learning rate, gradient clip 1.0.
- A new adapter for each dataset, created from the base, not from an old adapter. Save under a new directory per dataset. Do not write `/lora/adapter` or `/lora/adapter-pass2`.
- Do not merge.

Letter cross-entropy is the loss over the option letters only. The target is one-hot except where a dataset stores a soft teacher distribution. typed-decisions, if the file stores a single letter, stays one-hot on that letter. Say so in the manifest.

## One prompt, one index

Freeze this user line for every dataset, including typed-decisions:

```
A. {option text}
B. {option text}
```

The system line is the Phase 2 system line. `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`. Train and test call the same function.

Pad on the right in training and in scoring. Read the last real token, the last index whose attention mask is 1. Before any GPU job, assert that `mask.sum() - 1` equals that index on every row of a padded batch. Under right padding the two indexes are the same. If any row differs, stop. The left-pad exam index from Phase 0 is not used here.

The existing 1,289 of 2,000 is plain Gemma on the old exam prompt at the last real token. This protocol changes the prompt, so that 1,289 is not the before-count. Rescore plain Gemma on the frozen prompt first.

## Temperature

After the LoRA exists, fit T on that dataset’s calibration rows only. Grid 0.50 to 3.00 in steps of 0.05. Choose the T with the lowest mean letter NLL. Record accuracy at T = 1 and at the chosen T; they must be equal. typed-decisions calibration must contain yes/no, choice, and score rows, because the test contains all three. A missing type stays at T = 1 and the manifest says which type was missing.

## Order of work

Do not skip to training.

1. Write four manifests and four row files. Each manifest lists train, calibration, and test counts, the prompt hash, the padding side, and the label source. Tests are disjoint from train and calibration. This step is CPU only.
2. Score plain Gemma on each test. Four result files. Right padding. Last real token. No LoRA. T = 1.
3. Train four LoRAs, one at a time, each from the base. Modal L40S, `single_use_containers=True`, `min_containers=0`, `scaledown_window=2`. Chunk updates to about 8 minutes so the client is not cancelled. Each chunk starts a fresh AdamW. The learning-rate schedule uses the global step. Resume from that dataset’s own checkpoint.
4. Score each LoRA on its own test with the same code path as step 2.
5. Fit T per dataset on its calibration split. Rescore ECE at that T. Accuracy must match step 4.
6. Write one keep table. Four rows. A row is kept only when the correct count rose and ECE fell on that dataset’s test.

## What a result is allowed to say

A kept Banking77 adapter says the letter readout got more of Banking77’s official test right, and the percentages moved closer to the hit rate, under this prompt and this index. It does not say the adapter is a judge for a new bank’s private categories. Those categories need their own held-out tickets.

A kept typed-decisions adapter says the model moved closer to the teacher on the 2,000 held-out questions. It does not say the invoices are judged correctly beyond the teacher’s own agreement, about 1,470 of 2,000.

A miss on one dataset stays on that dataset. It does not retire the other three.
