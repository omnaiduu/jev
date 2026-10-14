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

The keep file is `results/phase6/keep.json`.

## Where to read the code

Start with `judge/`. That is the run behind the table: one adapter per dataset, the prompt, the keep rule.

`core/` is the shared code.

- `core/prompts.py` is the prompt. The letter is the slot. The words stay in the prompt.
- `core/loss.py` is the loss on those letters.
- `core/metrics.py` is accuracy, calibration error, and Brier score.
- `core/fit.py` fits one temperature per dataset.
- `core/score.py` reads the letter scores.

`phase0` to `phase5` are the earlier runs. They call `core`. The table above is not those runs.

`PROTOCOL.md` is why each dataset has its own adapter. `LEARNING.md` is the design. `posts/phase6-per-dataset/insights.md` is the measured argument.

## Checks

```bash
pip install -r requirements.txt
pytest judge/test_judge.py phase0/test_metrics.py phase1/test_phase1.py phase2/test_phase2.py phase3/test_phase3.py phase4/test_phase4.py phase5/test_phase5.py
```

These checks do not need a GPU. Training runs on Modal, and that image installs its own torch.

The `.jsonl` row files are not in git. The scores are the `.json` files under `results/phase6/`. Rebuild the rows with `judge/build.py` if you need them.
