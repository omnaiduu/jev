# System One on Gemma 4 E4B

One call takes a situation and a fixed list of answers, and returns a probability for each answer. The model is Gemma 4 E4B, text only. Each dataset has its own adapter. The adapters are not merged into Gemma.

The writeup is [One call on Gemma 4 E4B](https://omnaidu.com/blog/one-call-on-gemma-4-e4b).

## The result

Each count is correct answers on rows the adapter did not train on. The first count is the plain model. The second is the adapter.

| Dataset | Plain model | Adapter |
|---|---:|---:|
| BoolQ | 2,824 / 3,270 | 2,972 / 3,270 |
| Banking77 | 2,547 / 3,076 | 2,958 / 3,076 |
| MultiNLI | 7,393 / 9,815 | 8,984 / 9,815 |
| typed-decisions | 1,219 / 2,000 | 1,572 / 2,000 |

Banking77 shows 20 intents on each row, not 77. The typed-decisions before-count on this prompt is 1,219 / 2,000. SNLI was not trained on. With the MultiNLI adapter it is 6,140 / 9,842, then 8,909 / 9,842. That score does not decide whether the MultiNLI adapter is kept.

The keep file is `results/phase6/keep.json`. The checks are `phase6/test_phase6.py`.

## What is in this repo

- `phase0` through `phase6` is the code for each stage.
- `results/phase6` holds the scores.
- `PROTOCOL.md` is why each dataset has its own adapter.
- `LEARNING.md` is why the readout is a letter.
- `posts/phase6-per-dataset/insights.md` is the measured argument.

The older mixed-adapter comparison is not this result. Do not read 752 / 2,000 to 941 / 2,000, or 1,289 / 2,000 to 1,572 / 2,000, as the effect of these four adapters.
