"""Load GPU / vLLM / token-capture time series for dashboard charts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _epoch_to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ts, tz=UTC)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_gpu_log(path: Path) -> pd.DataFrame:
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        dt = _epoch_to_dt(row.get("logged_at") or row.get("sampled_at"))
        if dt is None:
            continue
        records.append(
            {
                "ts": dt,
                "gpu_util_pct": row.get("gpu_util_pct"),
                "mem_util_pct": row.get("mem_util_pct"),
                "mem_used_mib": row.get("mem_used_mib"),
                "mem_total_mib": row.get("mem_total_mib"),
                "temperature_c": row.get("temperature_c"),
                "power_w": row.get("power_w"),
                "sm_clock_mhz": row.get("sm_clock_mhz"),
                "gpu_name": row.get("gpu_name"),
            }
        )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    return df.sort_values("ts").reset_index(drop=True)


def load_vllm_metrics(path: Path) -> pd.DataFrame:
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        dt = _epoch_to_dt(row.get("logged_at"))
        if dt is None:
            continue
        records.append(
            {
                "ts": dt,
                "kv_used_frac": row.get("kv_used_frac"),
                "kv_used_tokens": row.get("kv_used_tokens"),
                "running": row.get("running"),
                "waiting": row.get("waiting"),
                "gen_tokens_total": row.get("gen_tokens_total"),
                "tokens_per_sec": row.get("tokens_per_sec"),
                "verdict": row.get("verdict"),
            }
        )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    return df.sort_values("ts").reset_index(drop=True)


def load_token_captures_light(path: Path) -> pd.DataFrame:
    """Latency / finish_reason / completion text — not token id dumps."""
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    records = []
    for i, row in enumerate(rows):
        started = row.get("request_started_at")
        created = row.get("created_at")
        dt = _epoch_to_dt(started) or _epoch_to_dt(created)
        text = row.get("text")
        if text is not None:
            text = str(text)
        records.append(
            {
                "index": i,
                "ts": dt,
                "request_id": row.get("request_id"),
                "model": row.get("model"),
                "latency_ms": row.get("latency_ms"),
                "finish_reason": row.get("finish_reason"),
                "text": text,
                "n_prompt_tokens": len(row.get("prompt_token_ids") or []),
                "n_completion_tokens": len(row.get("token_ids") or []),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    return df.sort_values("ts", na_position="last").reset_index(drop=True)


def filter_time_window(
    df: pd.DataFrame,
    start: datetime | None,
    end: datetime | None,
    *,
    pad_sec: float = 30.0,
) -> pd.DataFrame:
    if df.empty or "ts" not in df.columns:
        return df
    out = df
    if start is not None:
        start_utc = start if start.tzinfo else start.replace(tzinfo=UTC)
        out = out[out["ts"] >= start_utc - pd.Timedelta(seconds=pad_sec)]
    if end is not None:
        end_utc = end if end.tzinfo else end.replace(tzinfo=UTC)
        out = out[out["ts"] <= end_utc + pd.Timedelta(seconds=pad_sec)]
    return out.reset_index(drop=True)


def baseline_metric_paths(baseline_root: Path) -> dict[str, Path]:
    root = baseline_root.resolve()
    return {
        "gpu_log": root / "gpu_log.jsonl",
        "vllm_metrics": root / "vllm_metrics.jsonl",
        "token_captures": root / "captures" / "token_captures.jsonl",
    }
