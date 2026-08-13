"""A distillation run must be reconstructable after it ends.

Three files, three granularities: `opd_log.jsonl` per optimiser step,
`opd_examples.jsonl` per example, `opd_tokens.jsonl` per token. The step log
alone says a run went badly; only the token log says which tokens, what the
student wrote, and what the teacher wanted instead — the difference between a
number and a diagnosis.
"""

from __future__ import annotations

import inspect

from vektori_trace import distill


def _loop_source() -> str:
    return inspect.getsource(distill.run_opd_training)


def test_three_log_granularities_exist():
    src = _loop_source()
    for name in ("opd_log.jsonl", "opd_examples.jsonl", "opd_tokens.jsonl"):
        assert name in src, f"{name} is not written"


def test_step_log_schema_is_not_polluted():
    """Per-example and per-token records go to their own files.

    `opd_log.jsonl` is one line per optimiser step and has positional readers;
    interleaving other record types would silently change an artifact that
    already has consumers.
    """
    src = _loop_source()
    assert "ex_log.write(" in src
    assert "tok_log.write(" in src


def test_token_record_carries_both_sides_of_the_comparison():
    """The objective is a comparison; a log of one side cannot explain it."""
    src = _loop_source()
    for field in ("student_logprob", "teacher_logprob", "student_token"):
        assert f'"{field}"' in src, f"token log is missing {field}"


def test_token_record_carries_teacher_alternatives():
    """The teacher's top-K is what estimator A trains against. Without it you
    can see that the student was wrong but not what it should have said."""
    src = _loop_source()
    assert '"teacher_topk"' in src


def test_token_record_carries_coverage():
    """Mapped mass decides which estimator ran at each position.

    Below ~0.9 the span is demoted from A to B. If coverage silently degrades
    mid-run the objective quietly changes character, so it is logged per
    position rather than only measured in pre-flight.
    """
    src = _loop_source()
    assert '"mapped_mass"' in src
    assert '"mapped_count"' in src
    assert '"topk_width"' in src


def test_example_record_carries_what_the_student_wrote():
    """Numeric columns describe a decision whose content is otherwise
    unrecoverable. "It edited the workflow instead of the resolver" only reads
    out of the text."""
    src = _loop_source()
    assert '"action_text"' in src


def test_forensics_never_kill_a_run():
    """A logging bug must not destroy hours of GPU time. The token log writes
    its own failure as a record and continues."""
    src = _loop_source()
    assert "token_log_error" in src


def test_progress_reaches_stdout():
    """On Modal the JSONL files sit on container-local disk until the run ends,
    so `modal app logs` is the only live view of something billing by the
    second. Silence for two hours is not an acceptable view of it."""
    src = _loop_source()
    assert "flush=True" in src
    assert "step {step + 1" in src or "step %3d" in src


def test_token_log_is_cross_tokenizer_only():
    """The same-vocab path has no alignment to explain, and per-token records
    are the largest artifact a run produces."""
    src = _loop_source()
    assert "tok_log = tok_log_f if cfg.cross_tokenizer else None" in src
