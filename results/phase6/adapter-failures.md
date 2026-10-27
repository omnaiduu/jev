# Adapter failures

This CSV is the test questions the model got wrong after training. One row is one missed question. Right answers are not in the file.

1,675 rows: BoolQ 298, Banking77 118, MultiNLI 831, typed-decisions 428.

These match the published totals. BoolQ was 2,972 right out of 3,270, so 298 misses. Same check for the other three tests.

**Plain** means before the extra training. **Adapter** means after the extra training for that test.

## Two kinds of wrong

- **still_wrong**: wrong before training, and still wrong after.
- **lost**: right before training, then wrong after.

## Columns

| Column | Meaning |
| --- | --- |
| dataset | Which test: boolq, banking77, multinli, or typed-decisions. |
| id | The question’s id. |
| type | Kind of question. `noul` is yes/no. |
| outcome | `still_wrong` or `lost`. |
| question | The question. |
| state | The passage or customer message. |
| options | The choices, listed as A, B, C. On BoolQ, A is no and B is yes. |
| label | The stored right answer. |
| label_text | That answer in words. On BoolQ both say yes or no. On Banking77, `label` is the code and `label_text` is the words. |
| plain_answer | Answer code before training. |
| plain_answer_text | That answer in words. |
| plain_letter | Letter picked before training. A is the first choice. |
| plain_correct | `true` if that answer matched the label. |
| plain_confidence | How sure it was before training. 0.99 means 99%. |
| adapter_answer | Answer code after training. |
| adapter_answer_text | That answer in words. |
| adapter_letter | Letter picked after training. |
| adapter_correct | `false` on every row. |
| adapter_confidence_at_temperature_1 | How sure it was after training, before a small adjustment. |
| fitted_temperature | That adjustment. It changes the percentage only. The letter stays the same. BoolQ and Banking77 are 1.25, MultiNLI is 1.35, typed-decisions is 0.50. |
| adapter_confidence_at_fitted_temperature | How sure it was after the adjustment. |
| workflow | Filled on typed-decisions. Blank on the other tests. |
| question_name | Which question inside a typed-decisions case. Blank on the other tests. |
