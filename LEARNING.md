# Learning notes

This file is the “why” behind `PLAN.md`. Each section is a question that came up while designing the model, with the idea, a small example, and the choice we locked in.

The model we are building is a judge. You give it a situation and a closed list of answers. It returns a percentage for each answer. It does not write a paragraph.

Example:

```text
state: "I was charged twice. Please refund."
question: "Which team?"
options: billing, tech, sales
output: billing 0.91, tech 0.06, sales 0.03
```

Those three percentages are the whole product. Code can branch on them: auto-route if billing is above 0.85, otherwise ask a person.

## Instruction-tuned, and why the checkpoint name ends in `-it`

A raw language model is trained to continue text. Show it “the capital of France is” and it continues “Paris”. The raw Gemma checkpoint is `google/gemma-4-E4B`.

An **instruction-tuned** model had a second training pass on examples shaped like “a person asks, the assistant answers.” The checkpoint we use is `google/gemma-4-E4B-it`. The `-it` means instruction-tuned.

We use `-it` because our prompt is an instruction: here is a situation, here are the allowed answers. The chat template (the special tokens that mark the user turn and the assistant turn) matches how that checkpoint was trained. The raw checkpoint would treat that prompt as more text to continue, not as a question to score.

## Why text only, with vision and audio off

Gemma 4 E4B is multimodal. It has extra stacks that turn an image or a sound into tokens the text model can read. Our rows have neither. `state` is an email, a passage, or a short JSON record.

Phase 0 and the exam only need the text stack, because that stack is what produces a logit for the word `billing`. Running the vision or audio stack would spend GPU memory on an input we never send.

When we train, the LoRA is attached to the language, attention, and MLP layers only. Training the vision or audio weights would change parameters the text exam never reads. Unsloth’s own Gemma 4 notes say to leave `finetune_vision_layers` off until the data actually contains images.

## Logit, softmax, and logit readout

A **token** is a piece of text the model knows as one id. `billing` might be one token. A long word might be several.

A **logit** is the raw score the model gives that token at one position. It can be any number: 2.0, 0.0, −1.3.

**Softmax** turns a few logits into percentages that are positive and sum to 1.

```text
logits:    billing 2.0, tech 0.0, sales 0.0
softmax:   billing 0.79, tech 0.11, sales 0.11
```

A **logit readout** means: run one forward pass, take the logits at the last position, keep only the option tokens, softmax those. The model is not allowed to write a sentence. The percentages are the answer.

Phase 0 is this readout on plain E4B, before any LoRA. Those numbers are the bar. If training does not beat them on the same exam, the LoRA did not help.

The option token has to be one token. Gemma splits `human_review` into three pieces (`human`, `_`, `review`) and `harmful` into two (`harm`, `ful`). Those words have no single softmax slot. Phase 0 therefore binds each option to a letter, `A`, `B`, `C`, in the order the dataset lists them, and softmaxes the letter ids. The option id and its description stay in the prompt so the model still sees the words. `billing`, `true`, `false`, and the digits `0`–`3` are already one token, but the exam also contains the multi-token ids, so every question uses letters. Phase 4 has to use the same letters. Scoring the raw option-word logits would be a different exam.

## Why we train percentages instead of teaching the model to type a letter

The usual Unsloth path is **SFT** (supervised fine-tuning). The loss checks “was the next word the right word?” You would train the model to type `billing` or the letter `A`.

That teaches a pick. It does not teach that 0.91 should mean “right about 9 times in 10.” Our code will threshold on the percentage, so the training signal has to be the percentage list.

Unsloth still loads E4B and attaches the LoRA, because that is the memory-efficient loader. The loss itself is a short PyTorch loop, not `SFTTrainer`.

## Cross-entropy and KL

For a single correct option:

```text
loss = -log(percentage on the correct option)
```

| Percentage on the right option | Loss |
|---|---|
| 0.99 | 0.01 |
| 0.80 | 0.22 |
| 0.40 | 0.92 |
| 0.10 | 2.30 |

Correct and sure: tiny loss. Wrong and sure: large loss. Training steps the LoRA so this number gets smaller.

When the answer key is a spread, `target = [0.7, 0.2, 0.1]`, you do that once per option and add them, each weighted by the key:

```text
loss = -(0.7·log(p_billing) + 0.2·log(p_tech) + 0.1·log(p_sales))
```

That sum is still **cross-entropy**.

**KL divergence** is the same sum minus a piece that depends only on the answer key, not on the model. That extra piece does not change during training, so KL and cross-entropy move the weights the same way. For a one-hot key like `[1, 0, 0]` the extra piece is 0, and KL equals `-log(p_correct)`.

We use cross-entropy. If a row has a soft target, the weighted sum above is the loss for that row.

## LoRA, and why it stays a separate file

A **LoRA** is a thin set of extra weights clipped onto the frozen model. Rank 16 means each extra matrix is narrow. We train tens of millions of numbers instead of all of E4B.

We save the adapter as its own file and turn it on only for decisions. Merging it into Gemma changes the model you might also use for writing. On a similar Qwen setup, merging a decision adapter dropped coding accuracy by about 9 points even though the prose looked almost the same.

## What a training row is for

```json
{
  "state": "I was charged twice. Please refund.",
  "question": "Which team?",
  "options": ["billing", "tech", "sales"],
  "target": [1.0, 0.0, 0.0]
}
```

`target` is the answer key, in the same order as `options`. The row teaches one skill: given this text and this list of words, put the percentages in the right place. It does not teach the model to write the email.

Three piles, cut before training:

| Pile | What happens |
|---|---|
| train | The loss updates the LoRA. |
| calibration | The LoRA is frozen. Only the temperature is chosen here. |
| test | Never used to change anything. This is the published score. |

If test rows leak into training, the score measures memory. If you pick the best checkpoint by peeking at the test score, you have also used the test to choose.

## Open data and generated data

**Open datasets** already have labels made by people or by rules. We rewrite them into the row above.

- BoolQ becomes yes/no.
- MultiNLI becomes entail / neutral / contradict.
- Banking77 becomes “which support intent,” with the list kept to about 20 options.

This is the bulk of the train file. A public Qwen3.5-4B replica trained on about 38,000 questions of this kind and scored about 88% on a staging set it had not trained on.

**Generated rows** exist only for tasks those sets do not contain: your refund rule, “does this passage answer the question?” A script picks the truth first (`team = billing`). A larger model on Modal writes the email. The script writes `target`. The writer is good at prose. If the writer also filled `target`, the student would learn the writer’s mistakes.

Training on `jev-distill-corpus` is a different experiment. Those labels are Jev’s own percentages. The student learns to imitate Jev, including Jev’s mistakes. That can be a side run later. It is not the main train set, because the exam would then partly measure “how well did we clone Jev?”

## Why we shuffle options

Models prefer the first slot and the letter A. That habit is **position bias**. If `billing` is always option 1, the model can score well by picking slot 1.

Phase 0 measured it. On 2,000 exam questions the untouched model picked letter A 47.9% of the time. The correct answer is in that slot only 26.9% of the time. Accuracy is 0.376, just above a random 0.318, because a first-slot habit is a weak judge. The habit lives in the weights. Reading the logits only reports it. Training is what changes the weights. The shuffle below is how the rows teach the word instead of the slot.

A shuffled copy of the same fact:

```json
{ "options": ["sales", "billing", "tech"], "target": [0.0, 1.0, 0.0] }
```

Same email, same truth. The 1.0 moved with the word `billing`. After enough of these, the pattern that works in both orders is the meaning of the word.

We shuffle about 30% of train rows. At exam time we also ask once with the order flipped, and we count how often the pick changes. A flip means position bias is still there.

## ECE, the calibration file, and temperature

**ECE** (expected calibration error) is measured after training, on rows that did not update the weights. Take the percentage on the model’s top option. That percentage is its confidence. Put questions into buckets.

If the “about 80%” bucket has 100 questions and only 60 are actually right:

```text
gap = |0.80 - 0.60| = 0.20
```

ECE is the average of those gaps, weighted by how many questions fell in each bucket. ECE 0 means “when it says 80%, it is right about 80% of the time.” Accuracy can be high while ECE is bad: often right, and still saying 99% on the misses.

The **calibration file** is the calibration pile: a few thousand rows in the same format, never used to update the LoRA.

**Temperature** `T` is one positive number applied after training:

```text
percentages = softmax(logits / T)
```

| T | Effect |
|---|---|
| 1 | percentages unchanged |
| 2 | flatter, the top answer looks less sure |
| 0.5 | sharper, the top answer looks more sure |

Dividing every logit by the same `T` does not change which option wins. Accuracy stays the same. Only the stated percentages move, which is what ECE measures.

On the calibration pile, search for the `T` that makes `-log(percentage on the correct option)` as small as possible. One `T` per question type, because yes/no and 1-to-5 ratings are over-sure by different amounts. Save them next to the LoRA, for example `noul: 1.4`, `choice: 0.9`, `score: 1.8`. Fitting `T` on the train pile barely moves it: the model has memorized those rows and looks sure for the wrong reason.

Temperature cannot fix a wrong pick. A bad accuracy is a data or LoRA problem. A fine accuracy with a bad ECE is a temperature problem.

**Brier** score is one number that gets worse both when the pick is wrong and when the percentage was too extreme. We report it next to accuracy and ECE so a model cannot look good by winning only one of them.

## What the exam is, and why it is not our own data

We grade on `LocalLLaMA/typed-decisions`: 400 situations, 2,000 questions, written by someone else. We never train on it.

Grading on the emails we generated would be grading the practice sheet. A model can look perfect there and fail on a new source. A public result on a similar setup showed a new scoring head gaining on sources it had trained on and losing on sources it had never seen. That is why Phase 4 also scores one source left entirely out of training.

The typed-decisions labels come from another model that only agrees with itself about 73.5% of the time. That number is the ceiling. A score a little above it means we fit that teacher’s quirks, not that we solved judging.

## Latency

A normal chat reply has two clocks. **Prefill** reads the prompt in one forward pass. **Decode** writes the answer one token at a time, and each token is another step. A paragraph is dozens of extra steps. That second clock is what people feel as waiting.

This judge stops after prefill. The logits for A, B, C are the answer. Softmax over those few letters is free next to the forward pass. Decode never starts, so the reply has no length to wait on.

Training does not add a decode step. The same one pass still produces the percentages. The LoRA stays a separate file, so each text layer does one extra small multiply during that prefill. Temperature is a divide on a handful of logits after the pass.

We timed it on an L40S, batch size 1, median of 3 runs, thinking off. Same exam prompts. The container was single-use and is stopped. Numbers are in `results/phase0/latency.json`.

| prompt | input tokens | no decode | decode the letter | forced 32 new tokens |
|---|---|---|---|---|
| short yes/no | 176 | 65 ms | 131 ms | 1.94 s |
| mid choice | 375 | 65 ms | 134 ms | 1.95 s |
| longest in this exam | 685 | 78 ms | 144 ms | 2.10 s |

No decode is one forward pass. Decoding the letter still does that prefill, then writes 2 tokens (the letter, then stop), so it takes about twice as long. Forcing 32 new tokens, a short paragraph, takes about 2 seconds. Each generated token costs about 60 ms. From 176 tokens to 685, the prefill only moved from 65 ms to 78 ms. At these lengths the wait is the decode, not the prompt.

## Phase by phase, including the reason

### Phase 0 — Baseline

Load `google/gemma-4-E4B-it`. No LoRA. For each exam question, build the chat prompt, run one forward pass, softmax the letter logits. Save accuracy, ECE, and Brier.

Reason: without this file, a later score has nothing to beat. Training can look successful while the plain model would have scored the same.

What the run showed, on the 2,000-question test split, thinking off, no temperature:

| | accuracy | ECE | Brier |
|---|---|---|---|
| plain E4B | 0.376 | 0.366 | 0.905 |

A coin-flip over the real option counts would score 0.318. So the untuned model is only a little above chance. It is also over-sure: the average confidence on its top letter is 0.742. ECE is that gap. Phase 0 did not divide the logits by a temperature. Fitting `T` on the exam would leak the test into the dial. The calibration pile now exists, and `T` is still unfitted.

It prefers the first slot. Letter `A` won 47.9% of questions. The gold label is in slot 0 only 26.9% of the time. That gap is why this checkpoint needs training. The bias is in the weights, so a logit readout can see it and cannot remove it. Training on shuffled options moves the correct percentage with the words, so “always pick A” stops winning. Temperature only rescales confidence. The winning letter stays put. Phase 4 still flips the option order, to check the habit is actually gone.

Two hundred yes/no items have no `criteria` text. The gold label is still `false` or `true`, so the prompt uses a fixed pair of descriptions. Leaving those rows out would have scored 1,800 questions and broken the 2,000-question gate.

### Phase 1 — Data

Rewrite BoolQ, MultiNLI, and Banking77 into rows. Add a smaller generated set only where those three have no label. Shuffle about 30% of option orders. Cut train, calibration, and test before any weight update.

Reason: open labels teach general judgment. Generated rows teach rules nobody has labeled, with an answer key the script owns. The split keeps the exam honest.

What the cut produced:

| pile | rows |
|---|---|
| train | 40820 |
| calibration | 4000 |
| held out, SNLI only | 2000 |

30% of the train rows, 12246 of them, had their options shuffled, and the 1.0 moved with the correct word. Calibration was not shuffled. typed-decisions is not in either file.

Banking77 has 77 intents and no option list. Each row shows the true intent plus 19 others, in a random order. If the true intent were always first, the model could learn slot 0 again.

SNLI uses the same three labels as MultiNLI and is stored only in `held_out.jsonl`. Phase 4 scores it because the model never trained on that source. The public exam stays the test pile and was not copied in.

`google/gemma-4-12B-it` wrote the refund emails and the support notes. It is larger than E4B, and it never chose `target`. The script picked the team, or yes/no, before the writer ran. A row was kept only when the required phrase was actually in the text and the email did not name billing, tech, or sales.

### Phase 2 — Train

Unsloth loads E4B and attaches a rank-16 LoRA on the text layers. Vision and audio stay off. PyTorch applies cross-entropy. One pass over the train pile. Save the LoRA. Do not merge it.

Reason: SFT would teach the model to type a word. Cross-entropy teaches the percentage list. The separate LoRA file keeps normal Gemma intact.

### Phase 3 — Temperature

Freeze the LoRA. Fit one `T` per question type on the calibration pile only.

Reason: this is the ECE fix. It does not teach new facts. It rescales confidence so “80%” lines up with “right about 80% of the time.”

### Phase 4 — Exam

Score the LoRA on typed-decisions with `softmax(logits / T)`. Put the three numbers next to Phase 0. Flip the option order and count flips. Score one source that was never in the train pile.

Reason: the LoRA is worth keeping only if accuracy went up and ECE went down versus plain E4B, on data it was not trained on.

## Choices we are not taking in v1

| Other approach | Why it waits |
|---|---|
| No training at all, only the Phase 0 readout | We still run it. We only stop there if it already wins the exam. |
| SFT that types the letter `A` | The loss never sees the percentage list. |
| Train on Jev’s distill corpus as the main set | The labels are Jev’s answers, so the student clones Jev. |
| Merge the LoRA into Gemma | Writing quality of the base model drops. |
| A custom head and a 27B model, the Clef setup | Same API, much more work. E4B is the first experiment. |
| Use this model as an embedder | Embeddings need a different loss (pull similar texts together). A chat or judge checkpoint is a weak index. A small embedding model stays the retriever. |
