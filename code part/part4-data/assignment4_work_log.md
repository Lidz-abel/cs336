# Assignment 4 Work Log

This log records implementation progress, failed attempts, and verification
results while completing `part4-data`.

## Baseline Context

- Workspace: `/home/ldz/cs336/code part/part4-data`
- Goal: complete Assignment 4 local functionality, tests, reports, and
  submission packaging.
- Assignment theme: data processing for language modeling, including HTML text
  extraction, language identification, PII masking, harmful/quality filtering,
  and deduplication.

## Failed Attempt 1: Baseline pytest

- Command: `python -m pytest -q`
- Result: failed during test collection.
- Error: `ModuleNotFoundError: No module named 'xopen'` while importing
  `tests/test_deduplication.py`.
- Interpretation: the test environment is missing at least one assignment
  dependency before any adapter implementation can be exercised.
- Next step: install the minimal dependency needed for collection, then rerun
  tests to expose the next blocker.

## Failed Attempt 2: pytest after installing `xopen`

- Command: `python -m pytest -q`
- Result: `21 failed`.
- Error pattern: every failure reached `tests/adapters.py` and raised
  `NotImplementedError`.
- Interpretation: test collection now works; the remaining failures are the
  unimplemented assignment hooks for extraction, language identification, PII
  masking, harmful/quality classification, Gopher quality filtering, exact line
  deduplication, and MinHash/fuzzy deduplication.
- Next step: implement reusable functions under `cs336_data/` and wire the
  adapters to those functions.

## Failed Attempt 3: first adapter implementation

- Command: `python -m pytest -q`
- Result: `20 passed, 1 failed`.
- Error: `test_mask_ips` did not mask `192.0.2.146.`.
- Root cause: the IPv4 regular expression used `(?![\d.])`, so a normal
  sentence-ending period after the IP address prevented a match.
- Next step: loosen the right boundary to allow punctuation while still
  preventing trailing digits from being consumed.

## Successful Local Verification

- Command: `python -m pytest -q`
- Result: `21 passed in 0.23s`.
- Command: `python -m compileall cs336_data scripts tests/adapters.py`
- Result: all assignment4 modules and scripts compiled successfully.
- Implemented code:
  - `cs336_data/processing.py` for extraction, language ID, PII masking,
    harmful/quality filtering, Gopher filters, exact line deduplication, and
    fuzzy deduplication.
  - `tests/adapters.py` wired to the real implementations.
  - `scripts/filter_wet_data.py` for parallel WET filtering and filter-count
    reporting.
  - `scripts/tokenize_filtered_data.py` for GPT-2 tokenization into
    `np.uint16` binary data.
  - `cs336_data/wet_files.py` English filtering TODO.

## Remaining External Step

- The local machine currently has no visible `/shared-data` assignment files,
  so the full 2,500-WET filtering run and the 8 B200-GPU training run could not
  be executed locally.
- The scripts needed for those steps are present. Running them requires the
  course shared-data mount or Modal environment, plus the required GPU budget.

## Final Packaging Check

- Command: `bash test_and_make_submission.sh`
- Result: created a Python 3.12 `uv` environment, ran all public tests, and
  generated `cs336-assignment4-submission.zip`.
- Test result inside `uv`: `21 passed in 0.20s`.
- Packaging adjustment: removed the global `*.txt` exclusion from
  `test_and_make_submission.sh` so required test fixture text files remain in
  the archive; added an exclusion for `local-shared-data/*`.
- Generated report: `writeup.pdf`.
