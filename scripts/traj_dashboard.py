#!/usr/bin/env python3
"""Trajectory debug dashboard for harbor / passk baseline runs.

Run from the repo root:

    uv run --with streamlit --with plotly streamlit run scripts/traj_dashboard.py
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import *modules* (not symbols). Streamlit re-runs this script on edit but
# leaves packages already in sys.modules untouched — from-imports then bind
# stale LoadedTrajectory / load_trajectory and blow up on new fields.
from vektori_trace.dashboard import discover as discover_mod  # noqa: E402
from vektori_trace.dashboard import load_atif as load_atif_mod  # noqa: E402
from vektori_trace.dashboard import load_metrics as load_metrics_mod  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "qwen-run" / "vektori-out" / "baseline"


def _reload_dashboard_modules() -> None:
    """Reload dashboard packages so Streamlit hot-reload sees disk changes.

    Root cause of `LoadedTrajectory has no attribute steps_duration_sec`:
    UI script reloaded with code reading the new field, while load_atif stayed
    cached in sys.modules as the pre-change class/loader.
    """
    importlib.reload(load_atif_mod)
    importlib.reload(discover_mod)
    importlib.reload(load_metrics_mod)


def _ensure_state() -> None:
    if "expand_all" not in st.session_state:
        st.session_state.expand_all = False
    if "warnings_only" not in st.session_state:
        st.session_state.warnings_only = False
    if "jump_step" not in st.session_state:
        st.session_state.jump_step = 0


def _fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as e:
        return f"[unreadable: {e}]"


def _full_block(text: str, *, language: str | None = None) -> None:
    """Render full text with no truncation."""
    if not text:
        st.caption("(empty)")
        return
    st.code(text, language=language)


def _tool_calls_block(step: Any) -> None:
    if not step.tool_calls:
        st.caption("No tool calls")
        return
    for i, tc in enumerate(step.tool_calls):
        name = tc.get("function_name") or tc.get("name") or "tool"
        call_id = tc.get("tool_call_id") or tc.get("id") or f"#{i}"
        st.markdown(f"**{name}** (`{call_id}`)")
        args = tc.get("arguments")
        if isinstance(args, dict) and "keystrokes" in args:
            st.markdown("keystrokes:")
            _full_block(str(args.get("keystrokes") or ""), language="bash")
            extra = {k: v for k, v in args.items() if k != "keystrokes"}
            if extra:
                _full_block(json.dumps(extra, indent=2, ensure_ascii=False), language="json")
        else:
            _full_block(json.dumps(args, indent=2, ensure_ascii=False), language="json")


def _aware(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def render_trial_list(refs: list[Any]) -> None:
    rows = [
        {
            "status": str(r.status),
            "trial_id": r.trial_id,
            "task": r.task,
            "attempt": r.attempt,
            "steps": r.n_steps,
            "started_at": r.started_at,
            "model": r.model,
            "exception": r.exception_one_liner,
        }
        for r in refs
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_header(ref: Any, traj: Any | None) -> None:
    st.subheader(ref.trial_id)
    cols = st.columns(4)
    cols[0].metric("Status", str(ref.status))
    n_steps = len(traj.steps) if traj else (ref.n_steps or 0)
    cols[1].metric("Steps", n_steps)
    cols[2].metric("Wall clock", _fmt_duration(traj.duration_sec if traj else None))
    tokens = "—"
    if traj and traj.final_metrics:
        p = traj.final_metrics.get("total_prompt_tokens")
        c = traj.final_metrics.get("total_completion_tokens")
        if p is not None or c is not None:
            tokens = f"{p or 0} / {c or 0}"
    cols[3].metric("Tokens (prompt/completion)", tokens)

    st.caption(
        f"agent={ref.agent or (traj.agent_name if traj else None)} · "
        f"model={ref.model or (traj.model_name if traj else None)}"
    )
    if traj is not None:
        steps_dur = getattr(traj, "steps_duration_sec", None)
        steps_start = getattr(traj, "steps_started_at", None)
        steps_end = getattr(traj, "steps_ended_at", None)
        if steps_dur is None and traj.steps:
            first = traj.steps[0].timestamp_dt
            last = traj.steps[-1].timestamp_dt
            steps_start = steps_start or first
            steps_end = steps_end or last
            if first is not None and last is not None:
                steps_dur = (last - first).total_seconds()
        steps_span = _fmt_duration(steps_dur)
        wall_span = _fmt_duration(traj.duration_sec)
        note = (
            f"Wall clock {wall_span} "
            f"({traj.started_at} → {traj.ended_at}). "
            f"Agent steps in trajectory.json span {steps_span} "
            f"({steps_start} → {steps_end})."
        )
        if (
            steps_dur is not None
            and traj.duration_sec is not None
            and traj.duration_sec - steps_dur > 60
        ):
            note += (
                " Gap is usually summarization / JSON retries after the last "
                "dumped step (timeout kills before result.json)."
            )
        st.info(note)
    st.code(str(ref.trial_dir), language=None)
    if ref.exception_one_liner:
        st.warning(ref.exception_one_liner)


def render_steps(traj: Any) -> None:
    toolbar = st.columns([1, 1, 1, 2])
    if toolbar[0].button("Expand all", use_container_width=True):
        st.session_state.expand_all = True
    if toolbar[1].button("Collapse all", use_container_width=True):
        st.session_state.expand_all = False
    st.session_state.warnings_only = toolbar[2].checkbox(
        "Warnings only",
        value=st.session_state.warnings_only,
    )
    step_ids = [s.step_id for s in traj.steps]
    jump = toolbar[3].number_input(
        "Jump to step",
        min_value=min(step_ids) if step_ids else 0,
        max_value=max(step_ids) if step_ids else 0,
        value=st.session_state.jump_step or (step_ids[0] if step_ids else 0),
        step=1,
    )
    st.session_state.jump_step = int(jump)

    if traj.summarization_indices:
        st.info(
            "Proactive summarization episodes on this run: "
            + ", ".join(str(i) for i in traj.summarization_indices)
            + " (see Summarizations tab)"
        )

    expanded = bool(st.session_state.expand_all)
    steps = traj.steps
    if st.session_state.warnings_only:
        steps = [s for s in steps if s.has_json_warn]

    if not steps:
        st.warning("No steps to show with the current filter.")
        return

    for step in steps:
        force_open = expanded or step.step_id == st.session_state.jump_step
        delta = f"Δ{step.delta_sec:.1f}s" if step.delta_sec is not None else "Δ—"
        tok = ""
        if step.prompt_tokens is not None or step.completion_tokens is not None:
            tok = f" · tok {step.prompt_tokens or 0}/{step.completion_tokens or 0}"
        preview = step.first_command_preview or "(no command)"
        title = (
            f"#{step.step_id} · {step.source} · {step.timestamp or 'no-ts'} · "
            f"{delta}{tok} · {preview}"
        )
        badges = []
        if step.has_think:
            badges.append("think")
        if step.has_json_warn:
            badges.append("json_warn")
        if badges:
            title = f"{title} · [{' '.join(badges)}]"

        with st.expander(title, expanded=force_open):
            st.markdown("#### Message")
            _full_block(step.message)
            st.markdown("#### Tool calls")
            _tool_calls_block(step)
            st.markdown("#### Observation")
            _full_block(step.observation_text)


def render_charts(baseline_root: Path, traj: Any | None) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    paths = load_metrics_mod.baseline_metric_paths(baseline_root)
    gpu = load_metrics_mod.load_gpu_log(paths["gpu_log"])
    vllm = load_metrics_mod.load_vllm_metrics(paths["vllm_metrics"])

    start = _aware(traj.started_at) if traj else None
    end = _aware(traj.ended_at) if traj else None

    use_window = st.checkbox("Limit charts to trial time window", value=bool(traj))
    if use_window and traj:
        gpu = load_metrics_mod.filter_time_window(gpu, start, end)
        vllm = load_metrics_mod.filter_time_window(vllm, start, end)

    if gpu.empty and vllm.empty:
        st.warning(
            "No metric files found. Expected "
            f"`{paths['gpu_log']}` and/or `{paths['vllm_metrics']}`."
        )
        return

    step_ts = []
    if traj:
        for s in traj.steps:
            dt = _aware(s.timestamp_dt)
            if dt is not None:
                step_ts.append(dt)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("GPU util % / power W", "GPU memory MiB", "vLLM KV frac / tok/s"),
    )

    if not gpu.empty:
        fig.add_trace(
            go.Scatter(x=gpu["ts"], y=gpu["gpu_util_pct"], name="gpu_util_%", mode="lines"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=gpu["ts"], y=gpu["power_w"], name="power_W", mode="lines"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=gpu["ts"], y=gpu["mem_used_mib"], name="mem_used_MiB", mode="lines"),
            row=2,
            col=1,
        )

    if not vllm.empty:
        fig.add_trace(
            go.Scatter(x=vllm["ts"], y=vllm["kv_used_frac"], name="kv_used_frac", mode="lines"),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=vllm["ts"], y=vllm["tokens_per_sec"], name="tokens_per_sec", mode="lines"
            ),
            row=3,
            col=1,
        )

    for ts in step_ts:
        fig.add_vline(
            x=ts,
            line_width=1,
            line_dash="dot",
            line_color="rgba(120,120,120,0.35)",
        )

    fig.update_layout(height=820, legend=dict(orientation="h"), margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if traj and step_ts:
        st.caption(f"{len(step_ts)} step markers overlaid as vertical dotted lines.")


def render_summarizations(ref: Any, traj: Any | None) -> None:
    agent_dir = ref.trial_dir / "agent"
    indices = traj.summarization_indices if traj else []
    if not indices and agent_dir.is_dir():
        indices = load_atif_mod.list_summarization_indices(agent_dir)
    if not indices:
        st.info("No summarization episode files for this trial.")
        return

    for idx in indices:
        paths = load_atif_mod.load_summarization_episode(agent_dir, idx)
        st.markdown(f"### Summarization episode {idx}")
        if "summary" in paths:
            st.markdown("**Summary agent output (full)**")
            _full_block(load_atif_mod.summarization_agent_message(paths["summary"]))
        for kind in ("questions", "answers"):
            if kind not in paths:
                continue
            with st.expander(f"{kind} ATIF (full agent message)", expanded=False):
                _full_block(load_atif_mod.summarization_agent_message(paths[kind]))
            with st.expander(f"raw {paths[kind].name}", expanded=False):
                _full_block(_read_text(paths[kind]), language="json")


def render_raw_logs(ref: Any) -> None:
    trial_dir = ref.trial_dir
    files = [
        ("trial.log", trial_dir / "trial.log"),
        ("terminus_2.pane", trial_dir / "agent" / "terminus_2.pane"),
        ("config.json", trial_dir / "config.json"),
        ("result.json (trial)", trial_dir / "result.json"),
        ("result.json (job)", trial_dir.parent / "result.json"),
        ("exception.txt", trial_dir / "exception.txt"),
        ("lock.json", trial_dir / "lock.json"),
    ]
    for label, path in files:
        if not path.is_file():
            continue
        with st.expander(f"{label} — {path.name}", expanded=False):
            st.caption(str(path))
            lang = "json" if path.suffix == ".json" else None
            _full_block(_read_text(path), language=lang)


def render_captures(baseline_root: Path, traj: Any | None) -> None:
    import plotly.express as px

    path = load_metrics_mod.baseline_metric_paths(baseline_root)["token_captures"]
    df = load_metrics_mod.load_token_captures_light(path)
    if df.empty:
        st.warning(f"No captures at `{path}`")
        return

    start = _aware(traj.started_at) if traj else None
    end = _aware(traj.ended_at) if traj else None

    windowed = load_metrics_mod.filter_time_window(df, start, end) if traj else df
    st.metric("Capture lines in window", len(windowed))
    if "finish_reason" in windowed.columns and not windowed.empty:
        st.write(windowed["finish_reason"].value_counts(dropna=False))

    plot_df = windowed.dropna(subset=["ts", "latency_ms"]) if not windowed.empty else windowed
    if not plot_df.empty:
        fig = px.line(plot_df, x="ts", y="latency_ms", title="Request latency (ms)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Completion texts (full)")
    for _, row in windowed.iterrows():
        rid = row.get("request_id") or row.get("index")
        lat = row.get("latency_ms")
        fr = row.get("finish_reason")
        label = f"{rid} · {fr} · {lat:.0f} ms" if lat == lat else f"{rid} · {fr}"
        with st.expander(str(label), expanded=False):
            text = row.get("text")
            _full_block("" if text is None else str(text))
            if st.checkbox("Show capture metadata", key=f"raw_cap_{rid}"):
                meta = {
                    "request_id": row.get("request_id"),
                    "model": row.get("model"),
                    "latency_ms": row.get("latency_ms"),
                    "finish_reason": row.get("finish_reason"),
                    "n_prompt_tokens": row.get("n_prompt_tokens"),
                    "n_completion_tokens": row.get("n_completion_tokens"),
                    "ts": str(row.get("ts")),
                }
                _full_block(json.dumps(meta, indent=2), language="json")


def main() -> None:
    st.set_page_config(page_title="Trajectory Debug", layout="wide")
    _reload_dashboard_modules()
    _ensure_state()
    st.title("Trajectory debug dashboard")

    with st.sidebar:
        st.header("Data root")
        root_str = st.text_input("Baseline root", value=str(DEFAULT_ROOT))
        baseline_root = Path(root_str).expanduser()
        if not baseline_root.is_absolute():
            baseline_root = (REPO_ROOT / baseline_root).resolve()
        else:
            baseline_root = baseline_root.resolve()
        refresh = st.button("Rescan")

    if refresh:
        st.cache_data.clear()

    @st.cache_data(show_spinner=False)
    def _cached_discover(root: str, _loader_mtime: float) -> list[Any]:
        # _loader_mtime busts the cache when dashboard modules change on disk,
        # so we never serve TrialRef objects pickled under a previous class.
        return discover_mod.discover_trials(Path(root))

    if not baseline_root.is_dir():
        st.error(f"Baseline root does not exist: {baseline_root}")
        return

    loader_mtime = Path(load_atif_mod.__file__).stat().st_mtime
    refs = _cached_discover(str(baseline_root), loader_mtime)
    if not refs:
        st.warning(f"No trials found under `{baseline_root / 'passk_jobs'}`")
        return

    st.markdown("### Trials")
    render_trial_list(refs)

    labels = [
        f"{r.trial_id} · {r.status}"
        + (f" · attempt {r.attempt}" if r.attempt is not None else "")
        + (f" · {r.n_steps} steps" if r.n_steps else "")
        for r in refs
    ]
    default_i = discover_mod.default_trial_index(refs)
    selected_label = st.selectbox("Select trial", labels, index=default_i)
    ref = refs[labels.index(selected_label)]

    traj = None
    if ref.trajectory_path and ref.trajectory_path.is_file():
        try:
            traj = load_atif_mod.load_trajectory(ref.trajectory_path)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to load trajectory: {e}")

    render_header(ref, traj)

    tab_steps, tab_charts, tab_summ, tab_raw, tab_cap = st.tabs(
        ["Steps", "Metrics charts", "Summarizations", "Raw logs", "Captures"]
    )
    with tab_steps:
        if traj is None:
            st.info("No trajectory.json for this trial.")
            if ref.exception_one_liner:
                st.error(ref.exception_one_liner)
        else:
            render_steps(traj)
    with tab_charts:
        render_charts(baseline_root, traj)
    with tab_summ:
        render_summarizations(ref, traj)
    with tab_raw:
        render_raw_logs(ref)
    with tab_cap:
        render_captures(baseline_root, traj)


if __name__ == "__main__":
    main()
