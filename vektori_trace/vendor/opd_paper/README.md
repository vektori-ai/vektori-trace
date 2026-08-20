# Pinned reference implementation — *Breaking the Tokenizer Barrier*

Vendored verbatim for provenance. **Do not import from these files and do not
reformat them** (they are excluded from ruff). They exist so the port in
`vektori_trace/chunk_opd.py` can be diffed against the code it claims to
implement, which is what `docs/OPD-MULTITURN-PLAN.md` §5 requires:

> We will port or adapt the authors' released implementation rather than invent
> new clamps or credit assignment.

- Paper: Niu et al., arXiv:2606.09456
- Repo: https://github.com/ivanniu/On-Policy-Distill
- Revision: `927a8264f2e303b7f82c2d331a58fd4240c8805a` (see `REVISION`)

| vendored file | upstream path | sha256 |
| --- | --- | --- |
| `reward_manager_opd.py` | `verl/workers/reward_manager/opd.py` | `a24be172575818e3b44ac6287d68e9984a956c467f2e3baaa90d6c2ad365fe3e` |
| `core_algos.py` | `verl/trainer/ppo/core_algos.py` | `dd0d93e9ff039b36fa2ca565d730c4f30690dc5b04078ac3dfb28ba2c442ccd2` |

## What we port, and the one place we deliberately differ

`_align_chunks` (reward_manager_opd.py:330) and `compute_policy_loss_opd`
(core_algos.py:1035) are the two functions of record.

The upstream aligner **decodes token id ranges with each tokenizer** and
compares NFC-normalised strings, with U+FFFD ("incomplete") used to drive the
catch-up. Our `align.py` already aligns on **raw token bytes**, which makes the
same chunk boundaries reachable without a decode-per-candidate and without the
replacement-character heuristics — a byte run either matches or it does not.
`chunk_opd.align_chunks` therefore consumes `align.align_by_bytes` spans and
applies upstream's *credit assignment* unchanged.

Consequences kept identical to upstream:

- `large_chunk_threshold=6` — chunks exceeding it on either side become an
  `inf` sentinel, not a hard error, and get advantage 0.
- Unalignable tails are sentinels, never a hard-averaged guess.
- Degenerate `|L_S| < 1e-8` falls back to `L_T / n`, upstream's branch.
- `opd_loss_max_clamp` defaults to `None` (no clamp).

The plan's §6.5 "no silent `min_len` truncation" applies to *our* code paths;
upstream has no equivalent, so nothing is dropped in the port.
