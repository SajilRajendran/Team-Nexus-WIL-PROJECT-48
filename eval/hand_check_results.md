# Hand-checked evaluation results

**Run by:** Harshavardhen Gopal (Evaluation & Testing Lead)
**Verified by:** Sajil Rajendran
**Model:** Llama 3.1 8B via Ollama (temperature 0.1)

## Method
- Ten questions with a known expected answer: the nine known-answer seed questions (K01–K09) and one inferred question (I02).
- Expected answers were written in advance from the source documents (see `tests/eval_questions.csv`).
- **Correct answer:** states the expected fact without adding unsupported claims.
- **Correct citation:** the cited document actually contains the supporting passage (checked by opening the cited document).

## Answer correctness: 9 / 10 (90%)

| ID | Question (short) | Result |
|----|------------------|--------|
| K01 | VicSmart decision time | Correct |
| K02 | Are VicSmart applications advertised? | Correct |
| K03 | Who decides a VicSmart application? | Correct |
| K04 | What applications can use VicSmart? | Correct |
| K05 | How many centres have finalised plans? | Correct |
| K06 | How many new homes, by when? | Correct |
| K07 | Compulsory land acquisition? | **Incorrect** |
| K08 | Who builds the homes, over what time frame? | Correct |
| K09 | Which centres were in the Pilot Program? | Correct |
| I02 | Will my Ringwood house be compulsorily acquired? | Correct |

## Citation correctness: 3 / 4 (75%)

| ID | Citation result |
|----|-----------------|
| K03 | Correct |
| K04 | Correct |
| K09 | **Incorrect** – answer correct, but cited document did not contain the supporting passage |
| I02 | Correct |

## Notes
- Small sample: one question changes a percentage by 10 points.
- Results were recorded by hand; the full 22-question set should be re-run with `python -m eval.run_eval` and the output saved.
