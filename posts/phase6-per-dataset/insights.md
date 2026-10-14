# What the four LoRAs showed

12 October 2026. The measured argument for the public post. The keep file is `results/phase6/keep.json`. Every count below is from that file and the score files next to it.

The product is unchanged. A situation and a closed list of answers go in. One forward pass of Gemma 4 E4B returns a percentage per answer. The percentage is a softmax over the option letters. The model does not write a sentence.

Phase 6 trained four rank-16 LoRAs from the frozen base, one dataset each, and kept a LoRA only when its own test got more answers right and the stated percentage moved closer to the hit rate. All four were kept. SNLI rode along with the MultiNLI adapter and does not decide a keep.

## The opening a public post can use

Plain Gemma was already a strong judge on the closed lists it had seen in pretraining: 86% on BoolQ, 83% on a 20-way banking list, 75% on MultiNLI. One epoch of a small LoRA, trained on that dataset alone, beat each of those counts and made the percentages more honest. The same adapter, moved to SNLI, also learned to say “contradiction,” which the base almost never said. A fifth dataset, whose answers the base had not practiced, went from 61% to 79% agreement with a stored teacher label.

The temperature that makes those percentages honest is a different number on each dataset. Three of them needed to be softened. One needed to be sharpened, and the sharpening ran into the bottom of the grid.

## What the percentage is for

The winning letter is the decision. The percentage on that letter is how often a decision stated at that level is right. A product can handle a high percentage and hand a low percentage to a person. The split works when those two numbers meet. A percentage above the hit rate automates mistakes. A percentage below the hit rate sends finished work back to a person.

Phase 6 moved both numbers. The letter was right more often, and the gap between the stated percentage and the hit rate shrank. Most of that gap closed in the weights. Temperature was the smaller second turn of the knob: down for BoolQ, Banking77, and MultiNLI, up for typed-decisions.

BoolQ shows the before. Of 3,270 rows, 3,069 were stated above 90%, at an average of 99.5%, and were right 88.7% of the time. The remaining 201 rows were right 50.7% of the time. Almost-certain was wrong about 11 times in 100. Less-sure was a coin flip, so it was a weak reason to call a person.

After that LoRA, the 80–90% band is 311 rows, stated at 85.6%, right 80.4% of the time. The 60–70% band is 79 rows, stated at 65.7%, right 53.2% of the time. The high end became a percentage a threshold can use. The middle still cannot mean “probably wrong.”

Banking77 is the cleaner after. Its top band is 2,918 of 3,076 rows, stated at 99.3%, right 98.0% of the time. Before training the file was 2,547 / 3,076, and the stated percentages sat about 12 points above the hit rate. typed-decisions is the other loudness. The LoRA is right on 78.6% of rows and, before temperature, states about 61%. The plain model on that test stated about 91% while right 61% of the time.

Phase 6 did not time the forward pass. The only latency file is `results/phase0/latency.json`, plain Gemma on the old exam prompts, batch size 1, an L40S, median of three repeats: 64.5 ms, 65.4 ms, and 77.6 ms at 176, 375, and 685 tokens. Decoding the letter was 130.7 ms, 134.0 ms, and 143.5 ms. A forced 32-token paragraph was about 2 seconds. Those clocks were not retaken for the Phase 6 prompt or the four LoRAs.

## 1. The bar was the base, on this exact exam

A keep is two comparisons against the plain model on the same rows, the same prompt, and the same token.

| dataset | plain | LoRA | temperature | kept |
|---|---:|---:|---:|---|
| BoolQ | 2,824 / 3,270 | 2,972 / 3,270 | 1.25 | yes |
| Banking77 | 2,547 / 3,076 | 2,958 / 3,076 | 1.25 | yes |
| MultiNLI | 7,393 / 9,815 | 8,984 / 9,815 | 1.35 | yes |
| typed-decisions | 1,219 / 2,000 | 1,572 / 2,000 | 0.50 | yes |

`results/phase6/keep.json` has an empty `not_kept` list. The rule was harsh on purpose: the correct count has to rise, and calibration error at the fitted temperature has to fall. A model that gets more answers right and becomes more over-sure would have failed. None of these four failed.

The gains are 148, 411, 1,591, and 353. In accuracy that is 86.4% to 90.9%, 82.8% to 96.2%, 75.3% to 91.5%, and 61.0% to 78.6%.

## 2. The old mixed adapter is a different experiment

On 8 October a separate LoRA per dataset had not been trained. That is no longer true. Four directories exist: `/lora/v2-boolq`, `/lora/v2-banking77`, `/lora/v2-typed-decisions`, `/lora/v2-multinli`. None of them was loaded from `/lora/adapter` or `/lora/adapter-pass2`. None of them is merged into the base.

The mixed adapter is still the reason this run exists. At the last real token, on the old typed-decisions prompt, plain Gemma was 1,289 / 2,000 and that mixed adapter was 1,208 / 2,000. Training on BoolQ, MultiNLI, and Banking77 did not raise the invoice exam. Phase 6 then trained on the invoice questions themselves and scored 1,572 / 2,000.

Those two typed-decisions counts use different prompts. The old exam printed `A. id: text`. Phase 6 prints `A. {option text}` for every dataset, and the plain model on that prompt is 1,219 / 2,000. The keep uses 1,219 as the before-count. Publishing 1,289 to 1,572 as the effect of the new LoRA mixes two exams.

## 3. The readout had to be the token the loss trained

Gemma’s tokenizer pads on the left. The old exam index, `attention_mask.sum() - 1`, lands on the first real token of a left-padded row. Training had padded on the right, so the loss and the exam were reading different positions. On the old exam that disagreement changed the chosen letter on 824 of 2,000 LoRA rows and 1,228 of 2,000 plain rows.

Phase 6 builds the mask itself, pads on the right, and aborts the batch if those two indexes disagree. Padding asserts match two passes over each test (original order and reversed order): 6,540 on BoolQ, 6,152 on Banking77, 19,630 on MultiNLI, 4,000 on typed-decisions. Truncated rows are 0 on every score. The prompt hash is the same string in every manifest: `06bf695261cfc389a19f9b5b0476e0c9cb1d6384c98869068a6a49ea1d22b79b`.

A blog can say the percentage is read at the last real token, the same place the loss was computed. The old 752 / 2,000 to 941 / 2,000 pair is both models read at the early index. It is a different chart.

## 4. The letter is a slot. The words in the prompt are the skill

`human_review` is more than one Gemma token. A softmax over the vocabulary has no single cell for that string. Each answer is bound to A, B, C in the order printed, and the words sit in the prompt to the left of the position we read.

That is why the same three words can be two tasks. MultiNLI and SNLI both offer `entailment`, `neutral`, and `contradiction`. Plain Gemma scores 7,393 / 9,815 on MultiNLI and 6,140 / 9,842 on SNLI. On SNLI it chooses contradiction 367 times out of 3,278 labeled contradiction rows. Precision on those 367 calls is 0.989. Recall is 0.111. Neutral is chosen 5,894 times, precision 0.462. The base has a confident neutral prior on SNLI sentences and a working three-way judge on MultiNLI sentences. The label list is the same.

Banking77 says the same thing inside one dataset. The worst plain intent was `get_physical_card`, 4/40, and the substitutions were `passcode_forgotten` and `change_pin`. Nearby charge-names were confused with each other: `top_up_by_bank_transfer_charge` 15/40, `top_up_by_card_charge` 18/40. After the Banking77 LoRA, `get_physical_card` is 40/40, those two charge intents are 38/40, and perfect intents go from 10 to 21 of 77. The headline 2,547 to 2,958 hides the class that was actually broken.

## 5. Temperature has a sign

Fitting one scalar on the calibration rows, by letter negative log likelihood, grid 0.50 to 3.00:

| dataset | fitted T | what the probabilities were doing at T = 1 |
|---|---:|---|
| BoolQ | 1.25 | still a bit sharper than the hit rate |
| Banking77 | 1.25 | same shape, chosen on its own calibration cut |
| MultiNLI | 1.35 | same shape, a slightly larger scalar |
| typed-decisions | 0.50 | mean probability 0.611 at accuracy 0.786 |

BoolQ and Banking77 landing on 1.25 is two independent fits. It is a coincidence of this grid. typed-decisions went the other way, and 0.50 is the bottom of the grid. Calibration letter NLL there is 0.596 at `T = 1` and 0.453 at `T = 0.50`. A lower scalar might still help. This protocol did not extend the grid.

Accuracy at the fitted T equals accuracy at T = 1 on every dataset. Dividing every letter logit by the same positive number leaves the winning letter where it was. The keep uses the correct count from T = 1 and the calibration error from the fitted T.

A single temperature for the product would soften three adapters that were slightly sharp and would also soften the one adapter that was too diffuse. The grid caught the sign because each dataset was fit alone.

## 6. On the one-hot tasks, the weights did the calibration

Calibration error, plain at T = 1, then the LoRA at T = 1, then the LoRA at its fitted T:

| dataset | plain | LoRA at T = 1 | LoRA at fitted T |
|---|---:|---:|---:|
| BoolQ | 0.117 | 0.034 | 0.014 |
| Banking77 | 0.120 | 0.019 | 0.007 |
| MultiNLI | 0.183 | 0.029 | 0.010 |
| typed-decisions | 0.303 | 0.175 | 0.065 |

On BoolQ the total drop is 0.103, and 0.083 of it is already present before temperature. On Banking77, 0.101 of 0.113. On MultiNLI, 0.154 of 0.173. Temperature is the last cut.

typed-decisions splits the drop. The weights take it from 0.303 to 0.175. Sharpening at 0.50 takes it from 0.175 to 0.065. The soft training target is why. The loss asks the model to match a spread-out teacher vector, so the mode can be right while the stated percentage stays modest. Mean top-letter probability 0.611 against accuracy 0.786 is that shape. No LoRA row on that test puts more than 0.99 on the chosen letter.

The plain typed-decisions model had the opposite shape: mean probability 0.912 against accuracy 0.610.

## 7. The correct count can hide a trade. These runs mostly did not make one

BoolQ’s +148 is yes-recall. True positives on `yes` went from 1,722 to 1,871. True positives on `no` went from 1,102 to 1,101. Recall on `no` stayed at 0.89. Of the plain misses, 311 were a `no` on a `yes` passage. After the update, 162. The high learning rate did not turn the file into a yes-machine.

MultiNLI’s +1,591 raised every letter:

| letter | plain recall | LoRA recall | plain precision | LoRA precision |
|---|---:|---:|---:|---:|
| contradiction | 0.705 | 0.945 | 0.941 | 0.945 |
| entailment | 0.885 | 0.910 | 0.743 | 0.930 |
| neutral | 0.656 | 0.891 | 0.627 | 0.870 |

Contradiction true positives went from 2,265 to 3,035. Entailment false positives went from 1,064 to 238. The gain is spread across the three labels.

typed-decisions yes/no is the cleanest prior to watch. Of 600 yes/no rows, 305 are labeled `true`. The plain model said `true` on 517 of them and was right 376 times. The LoRA says `true` 287 times and `false` 313 times, and is right 516 / 600. The calls now sit on the label base rate.

By question type, plain then LoRA: choice 360 / 600 to 445 / 600, yes/no 376 / 600 to 516 / 600, score 483 / 800 to 611 / 800.

## 8. Beating 1,470 was allowed

A fresh sample from the typed-decisions teacher matches the stored label about 1,470 / 2,000 times. That is how often a draw at the teacher’s own temperature hits the mode. The LoRA scored 1,572 / 2,000 by agreeing with the stored mode more often than a fresh sample does.

The file allows that. A spread vector has a mode, and a sample sometimes misses it. We checked the stored label against the mode: `stored_label_not_mode` is 0. Ninety-six rows tie at that mode, and the tie does not move the stored label.

There is a second score on the same file. Soft negative log likelihood against the stored teacher vector, on the test, is 0.844. The entropy of those vectors averages 0.767. The student is close to the teacher’s distribution and a little worse than an exact copy. The keep bit uses only the mode count. A blog that publishes 1,572 should publish 0.844 beside it, and should publish 1,470 as a sample rate.

The training floor is that entropy. On the 4,000 train rows it averages 0.750. A printed loss near 0.75 means the batch matches the teacher. The last 100 batches of the final chunk averaged 0.827. The final printed batch, 0.657, is one batch under the floor. It can be a yes/no batch, whose own entropy is 0.449, or a batch the model made sharper than the teacher. One print does not say which.

## 9. SNLI was the transfer the mixed pile was hoping for

The MultiNLI LoRA, scored on SNLI with MultiNLI’s temperature of 1.35, and marked `enters_keep: false`:

| | correct | ECE |
|---|---:|---:|
| plain, T = 1 | 6,140 / 9,842 | 0.349 |
| LoRA, T = 1 | 8,909 / 9,842 | 0.030 |
| LoRA, T = 1.35 | 8,909 / 9,842 | 0.008 |

Contradiction recall went from 0.111 to 0.905. The model now chooses contradiction 3,112 times, against a support of 3,278. Neutral calls fall from 5,894 to 3,389. Row by row: 5,802 right before and after, 3,107 fixed, 338 broken, 595 wrong both times.

Before this adapter existed, the plain bins predicted a disappointment. MultiNLI’s top bin had a gap of 0.176 (probability 0.984, hit rate 0.808). SNLI’s top bin had a gap of 0.359 (probability 0.991, hit rate 0.632), and that bin held 8,985 of 9,842 rows. A temperature sized for the milder gap looked too small to fix SNLI. After the adapter, SNLI’s top bin at T = 1 is 7,866 rows, hit rate 0.961, probability 0.982, gap 0.022. The weights had already moved the mass. Temperature trimmed it.

That is the transfer result worth the post. Same three words, new sentences, a large move in the letter the base withheld. It stays outside the keep bit because SNLI never updated these weights, and the protocol promised not to let a transfer score decide the MultiNLI row.

## 10. When the model is unsure, that unsure-ness has to rank the hits

Plain BoolQ puts 3,069 of 3,270 rows above probability 0.9, hit rate 0.887, mean probability 0.995. The 201 rows below 0.9 have a hit rate of 0.507. Inside 0.5 to 0.9, accuracy sits near one half while the stated probability climbs. The 86% headline is the top bin, diluted by a coin flip. A temperature can pull 0.995 toward 0.86. It cannot make the middle bins start ranking hits, because those rows were not ordered by accuracy.

After the BoolQ LoRA the 0.8–0.9 bin is 311 rows, hit rate 0.804, probability 0.856. The ordering showed up in the weights. The 0.6–0.7 bin is still weak: 79 rows, hit rate 0.532 at probability 0.657. One scalar finished the dominant bin. The middle of the scale is only partly repaired.

Banking77 went the other way on sharpness and still calibrated, because the hits arrived with the confidence. Mean letter NLL fell from 1.109 to 0.140. The top bin is 2,918 / 3,076 rows, hit rate 0.980, mean probability 0.993. The plain model was sure and wrong on the rare intents. This one is sure and right on 96% of the file. `log(20)` is 2.996, so the plain model was already far above a uniform 20-way guess. The LoRA’s job was the leftover confusions, and it did them.

## 11. Order sensitivity is a margin, and a count

Reversing the option lines and asking whether the chosen option id changes:

| dataset | plain flips | LoRA flips | what happened to the margin |
|---|---:|---:|---|
| BoolQ | 86 / 3,270 | 78 / 3,270 | median gap 0.60 to 0.19 |
| Banking77 | 400 / 3,076 | 40 / 3,076 | remaining flips still sure, median gap 0.55 |
| MultiNLI | 1,051 / 9,815 | 65 / 9,815 | median gap 0.60 to 0.12 |
| SNLI | 727 / 9,842 | 77 / 9,842 | median gap 0.81 to 0.12 |
| typed-decisions | 305 / 2,000 | 187 / 2,000 | median gap 0.70 to 0.05 |

BoolQ’s flip count barely moved. The reversals that remain used to be sure and are now close. Banking77’s count collapsed and the few that remain are still sure. typed-decisions’ reversals became near-ties, which is what a soft letter distribution does when the lines swap. The keep bit ignores all of this. A blog that only prints the flip count will miss the BoolQ story, and a blog that only prints the margin will miss the Banking77 story.

## 12. Every kept adapter still breaks some of the base’s hits

| dataset | plain hits the LoRA broke | plain misses it fixed | share of plain hits broken |
|---|---:|---:|---:|
| BoolQ | 84 | 232 | 84 / 2,824 |
| Banking77 | 26 | 437 | 26 / 2,547 |
| typed-decisions | 131 | 484 | 131 / 1,219 |
| MultiNLI | 399 | 1,990 | 399 / 7,393 |

The short one-hot tasks spent little. Banking77 broke about 1% of the plain hits. BoolQ broke about 3%. MultiNLI, about fifty times as many steps at the same learning rate, broke about 5%. typed-decisions, the soft-target task, broke about 11%.

The net is still the keep on every row. The residue is the comparison a later lower-rate run would be judged against. On MultiNLI that residue is 399 rows the base already had right.

## 13. A high learning rate did not walk off a strong base

The worry, from the LoRA-versus-full-fine-tuning work, was that `2e-4` adds update directions that track forgetting when the base is already good. BoolQ was the obvious place: 929 steps, base at 86%. It gained 148 and held `no` recall. Banking77 was 999 steps, base at 83% on 20 ways. It gained 411, including the intent that looked least like a near-duplicate.

MultiNLI was the run the warning was about: 48,822 steps, base at 75%. The test moved to 8,984 / 9,815 and calibration error fell. The printed batch losses never settled into a falling curve. Chunk print-means sat roughly between 0.20 and 0.54 for the whole epoch, around a uniform three-way loss of `log(3) = 1.099`. The test count is the measurement. A print near 0.3 was one batch of eight.

A later BoolQ or MultiNLI run at `2e-5`, stopped when calibration loss stops falling, is still worth doing. It would be a comparison against a full epoch that already cleared the keep. It would be a new training run. These four adapters stay as they are.

## 14. The training log is one batch, and the checkpoint is the save

The trainer prints one batch of eight every 25 steps. On BoolQ, one confident miss averaged with seven easy rows prints about 0.70. A batch with no miss prints near 0. The same model produces both lines.

MultiNLI made that concrete for sixty chunks. Chunk 33 saved a last-batch loss of 1.536 while its printed batches averaged 0.222. Chunk 60, the end of the epoch, printed a last batch of 0.068 on the final 3 rows, while the 31 prints in that chunk averaged 0.332 and the last 100 batches averaged 0.270. The easy last batch is the save. It is a bad summary of the chunk.

The epoch was saved in chunks so a stopped client could resume from the last finished step. The learning rate follows the global step across those resumes. The scientific result does not depend on how many client processes it took.

## 15. Lines worth lifting

- Train one dataset, score that dataset, and write the plain count down first. The mixed pile could not tell “failed to learn” from “failed to transfer.”
- The base was already at 86%, 83%, and 75% on BoolQ, Banking77, and MultiNLI. The LoRAs beat those counts.
- Read the percentage at the last real token. The old exam index was a different letter.
- The letter is a slot. `get_physical_card` going from 4/40 to 40/40 is the result the total 2,547 to 2,958 was hiding.
- Three datasets wanted a temperature above 1. typed-decisions wanted 0.50, the bottom of the grid, because it was under-sure.
- On the one-hot tasks, most of the calibration happened in the weights. Temperature was the last cut. On the soft-target task, sharpening did about half.
- The percentage is what a product can branch on. Plain BoolQ said 99.5% on 3,069 of 3,270 rows and was right 88.7% of the time. After the LoRA, an 80–90% band of 311 rows is right 80.4% of the time. Banking77’s top band is 2,918 of 3,076 rows, stated at 99.3%, right 98.0% of the time. The 60–70% BoolQ band is still near a coin flip.
- 1,572 / 2,000 can sit above a 1,470 / 2,000 teacher-sample rate. The 1,470 is how often a fresh sample hits the mode.
- Same three words, two datasets. Plain SNLI almost never said contradiction. The MultiNLI adapter brought that recall from 0.111 to 0.905, and that score stays outside the keep.
- Report the letters. BoolQ’s gain was `yes` recall with `no` recall held. MultiNLI raised all three.
- A kept adapter still breaks some of the base’s hits: 84, 26, 131, and 399 on the four tests.
- A printed training loss is one batch. The MultiNLI prints jumped for the whole epoch. The test count moved by 1,591.

## 16. Lines that would be wrong in the post

- That 752 / 2,000 to 941 / 2,000 is the effect of training. Both of those reads are the early index.
- That 1,289 / 2,000 to 1,572 / 2,000 is this LoRA’s effect. The prompts differ. The before-count is 1,219 / 2,000.
- That 1,470 / 2,000 is a cap the student cannot pass.
- That one temperature, 1.25 or 1.30 or anything else, belongs on all four adapters.
- That the four adapters were merged, or that they should be added together. A single product adapter is a new training run and a new score on every dataset.
- That SNLI’s 8,909 / 9,842 is a fifth keep. It is a transfer check.
- That the old 1,820 / 2,000 SNLI sample is the same exam as 6,140 / 9,842. Different denominator, different prompt history.
- That a falling training print means the test moved, or that a noisy print means it did not.
- That extending the typed-decisions grid below 0.50, or rerunning BoolQ at `2e-5`, has already been done. Both are written down. Neither has been run.
- A Phase 6 latency. It was not recorded. The 65–78 ms forward pass is the Phase 0 file, on the old prompts, batch size 1.
- That every adapter became more sure. Three were softened. typed-decisions was sharpened.

## 17. What the post can point at next

These are the open experiments. They are the sequel, and the current post can name them in a last paragraph.

- Repeat BoolQ at learning rate `2e-5` and stop when calibration letter NLL stops falling. Score it with the same per-letter counts, against 2,972 / 3,270 and against the 84 broken hits.
- The same comparison is available for MultiNLI, against 8,984 / 9,815 and against the 399 broken hits. That run is about fifty times the steps. It is the expensive one.
- Extend the typed-decisions temperature grid below 0.50, fit on the calibration rows only, and report whether calibration NLL still falls. The winning letter will stay put.
- Divide the stored teacher vector by a teacher temperature before the typed-decisions loss, fit that choice on calibration only, and see whether test calibration moves. The keep count would still be agreement with the stored mode.

The four adapters, the four keeps, and the SNLI transfer are the piece that can be published now.
