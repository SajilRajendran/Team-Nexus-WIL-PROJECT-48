# Prompt and generation design

**Owner:** Karthic Raj Ayyappan (LLM / Generation Engineer)

## Model
- Llama 3.1 8B, run locally through Ollama (`llama3.2:3b` tested as a faster option).
- Temperature **0.1** for stable, repeatable answers. Context window 4096 tokens.

## Prompt rules (see `src/llm.py`)
1. Answer **only** from the numbered passages provided.
2. Cite the passages used with square brackets, e.g. [1] or [1][3].
3. If the passages do not contain the answer, reply with exactly `NOT_FOUND`.
4. Keep answers short and in plain English for residents.

## Two abstention gates
- **Gate 1 (before the model):** refuse if retrieval evidence is weak. The planned similarity threshold (0.45) refused no unanswerable questions in our sweep, so Gate 1 now uses the **re-rank score < 0**.
- **Gate 2 (after the model):** if the model replies `NOT_FOUND`, the app shows "I couldn't find this in the source documents."

## Fallback
If Ollama is not running, the app returns the most relevant source sentences (extractive fallback) instead of failing.

## How it works (four steps)
**Collect** official documents → **Retrieve** the best passages (hybrid search + re-ranking) → **Generate** a cited answer → **Respond** or refuse.
