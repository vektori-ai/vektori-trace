# Reference implementation — *Breaking the Tokenizer Barrier* (arXiv:2606.09456)

Upstream source for the cross-tokenizer OPD loss, kept whole and unmodified so
the port in `chunk_opd.py` can be diffed against the code it implements.

- Paper: Niu et al., arXiv:2606.09456
- Repo: https://github.com/ivanniu/On-Policy-Distill
- Revision: `927a8264f2e303b7f82c2d331a58fd4240c8805a` (see `REVISION`)

| file | upstream path | sha256 |
| --- | --- | --- |
| `reward_manager_opd.py` | `verl/workers/reward_manager/opd.py` | `a24be172575818e3b44ac6287d68e9984a956c467f2e3baaa90d6c2ad365fe3e` |
| `core_algos.py` | `verl/trainer/ppo/core_algos.py` | `dd0d93e9ff039b36fa2ca565d730c4f30690dc5b04078ac3dfb28ba2c442ccd2` |

The two functions of record are `_align_chunks` (`reward_manager_opd.py:330`)
and `compute_policy_loss_opd` (`core_algos.py:1035`). The files are kept whole
rather than excerpted: they carry verl framework context, and trimming them to
"just the relevant parts" risks dropping a helper the logic depends on, which
would silently break the ability to verify the port later.

`opd_manifest.verify_vendor_pins()` re-hashes both, so an edit here shows up as
a failed pin. Do not reformat them.

## What we port, and where we differ

The upstream aligner decodes token-id ranges with each tokenizer and compares
NFC-normalised strings, using U+FFFD to drive the catch-up. Our `align.py`
aligns on raw token **bytes**, reaching the same chunk boundaries without a
decode per candidate and without the replacement-character heuristics.
`chunk_opd.assign_chunk_advantages` consumes those byte-aligned spans and
applies upstream's credit assignment unchanged.

Kept identical to upstream:

- `large_chunk_threshold=6` — over-long chunks become a sentinel with advantage
  0, not a hard error.
- Unalignable tails are sentinels, never a hard-averaged guess.
- Degenerate `|L_S| < 1e-8` falls back to `L_T / n`.
- `opd_loss_max_clamp` defaults to `None` (no clamp).
