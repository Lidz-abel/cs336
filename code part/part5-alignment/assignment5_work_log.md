# Assignment 5 Work Log

This log records implementation progress, failed attempts, and verification
results while completing `part5-alignment`.

## Baseline Context

- Workspace: `/home/ldz/cs336/code part/part5-alignment`
- Goal: complete Assignment 5 local functionality, tests, reports, and
  submission packaging.
- Assignment theme: alignment and reasoning RL, including prompt/output
  tokenization, response log probabilities, rollout rewards, GRPO variants,
  off-policy importance weighting, SFT packing, parsing helpers, and DPO loss.

## Failed Attempt 1: Baseline pytest

- Command: `python -m pytest -q`
- Result: `26 failed`.
- Error pattern: every failure reached `tests/adapters.py` and raised
  `NotImplementedError`.
- Interpretation: the local environment can import torch, transformers, local
  tiny models, tokenizers, and snapshots. The remaining failures are the
  unimplemented assignment hooks.
- Next step: implement reusable alignment/RL helpers under `cs336_alignment/`
  and wire `tests/adapters.py` to those implementations.

## Failed Attempt 2: First implementation pass

- Command: `python -m pytest -q`
- Result: `23 passed, 3 failed`.
- Failed tests:
  - `test_tokenize_prompt_and_output`
  - `test_packed_sft_dataset`
  - `test_per_instance_dpo_loss`
- Diagnosis:
  - Prompt/output tokenization incorrectly used `full_ids[:-1]` for every
    sequence. The expected padded `input_ids` keep tokens up to the global
    `max_len`, while `labels` are shifted by one position.
  - Packed SFT examples missed an explicit EOS boundary between formatted
    instruction-response examples.
  - DPO initially used the raw prompt. The supplement test expects the Alpaca
    instruction template and includes EOS in the response likelihood.
- Fix:
  - Align `input_ids = full_ids[:max_len]`, `labels = full_ids[1:max_len+1]`,
    and mask only real response labels.
  - Strip formatted SFT examples and append `tokenizer.eos_token_id`.
  - Format DPO prompts with `prompts_safety/alpaca_sft.prompt` and append the
    tokenizer EOS token to chosen/rejected responses.

## Failed Attempt 3: Submission script in sandbox

- Command: `bash test_and_make_submission.sh`
- Result: script produced `code.zip`, but `uv run pytest` failed before tests.
- Error: `uv` tried to create files in `/home/ldz/.cache/uv`, which is read-only
  in this workspace.
- Fix: update `test_and_make_submission.sh` to default `UV_CACHE_DIR` to a local
  `.uv-cache` directory and exclude `.uv-cache/*` from the submission archive.

## Failed Attempt 4: uv dependency resolution without network

- Command: `UV_CACHE_DIR=.uv-cache uv run pytest -q`
- Result: failed while fetching the `flash-attn` wheel metadata from GitHub.
- Error pattern: network sandbox denied the connection.
- Fix: reran the same `uv` commands with network approval. After verification,
  removed the generated `.venv` and `.uv-cache` to recover disk space.

## Final Verification

- `python -m pytest -q`: `26 passed in 2.80s`.
- `UV_CACHE_DIR=.uv-cache uv run pytest -q`: `26 passed in 3.10s` with network
  approval.
- `UV_CACHE_DIR=.uv-cache bash test_and_make_submission.sh`: `19 passed` for the
  required GRPO tests and regenerated `test_results.xml` plus `code.zip`.
- Submission archive check: `code.zip` does not include `.venv`, `.uv-cache`, or
  nested `code.zip`.
- Disk cleanup: temporary `.venv` and `.uv-cache` were removed; workspace size
  returned to about `149M`, with about `12G` available on the filesystem.
