"""Put the repo root on `sys.path` for every pytest invocation.

`scripts/` is deliberately not a package — those files are entry points, run as
`python scripts/foo.py`. But the tests import them (`from scripts import
sft_stage_b_train_modal`), and pytest's default `prepend` import mode only adds
each test file's own directory. Whole-directory runs happened to work because
some other test module inserts the root at import time and collection order put
it first — that is luck, not configuration, and it made
`pytest tests/test_sft_stage_b_trainer.py` fail while `pytest tests/` passed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
