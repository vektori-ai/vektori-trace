"""Compaction evidence probe: marker -> main-traj -> sidecar -> prefix_turns_through_step."""
import json, hashlib, collections
from pathlib import Path
from vektori_trace.mining.atif import parse_job_trajectory
from vektori_trace.evaluate.resume import assistant_tool_steps
from vektori_trace.reopd import prefix_turns_through_step

TRIAL = Path("/data/vektori-out/dsv4-corpus60/passk_jobs/stage1/pallets__click-3704-1/"
             "pallets__click-3704-terminus-2/2026-08-14__03-13-38/pallets__click-3704__sCRM7w4")
out = {"trial": str(TRIAL)}

raw = json.load(open(TRIAL / "agent" / "trajectory.json"))
rsteps = raw["steps"]
out["raw_n_steps"] = len(rsteps)

# 1. markers in the RAW trajectory
markers = []
for i, s in enumerate(rsteps):
    cm = (s.get("extra") or {}).get("context_management")
    if cm:
        refs = []
        obs = s.get("observation") or {}
        for r in (obs.get("results") or []):
            for ref in (r.get("subagent_trajectory_ref") or []):
                refs.append(ref.get("trajectory_path"))
        markers.append({"raw_list_pos": i, "raw_step_id": s.get("step_id"),
                        "cm": cm, "sidecars": refs})
out["raw_markers"] = markers

# 2. parsed turns
turns = parse_job_trajectory(TRIAL)
steps = assistant_tool_steps(turns)
out["parsed_n_turns"] = len(turns)
out["parsed_n_steps_depth0"] = len(steps)
out["parsed_depths"] = dict(collections.Counter(t.subagent_depth for t in turns))

# 3. locate marker turns in parsed stream + map to replay step_index
marker_turns = [t for t in turns
                if t.role == "system" and "context summarization" in (t.content or "").lower()]
mm = []
for t in marker_turns:
    after = [i for i, (ti, _) in enumerate(steps) if ti > t.index]
    mm.append({"turn_index": t.index, "depth": t.subagent_depth,
               "first_replay_step_after": after[0] if after else None})
out["marker_turns"] = mm

# 4. THE question: does the prefix at the first post-compaction step still
#    contain pre-compaction content, and does it contain sidecar content?
d0 = [m for m in mm if m["depth"] == 0 and m["first_replay_step_after"] is not None]
if d0:
    T = d0[0]["first_replay_step_after"]
    mt = d0[0]["turn_index"]
    pre = prefix_turns_through_step(turns, T)
    out["probe_step_T"] = T
    out["prefix_n_turns"] = len(pre)
    out["prefix_turns_before_marker"] = sum(1 for t in pre if t.index < mt)
    out["prefix_turns_after_marker"] = sum(1 for t in pre if t.index > mt)
    out["prefix_depth_counts"] = dict(collections.Counter(t.subagent_depth for t in pre))
    out["prefix_chars_total"] = sum(len(t.content or "") for t in pre)
    out["prefix_chars_before_marker"] = sum(len(t.content or "") for t in pre if t.index < mt)
    out["prefix_chars_depth1"] = sum(len(t.content or "") for t in pre if t.subagent_depth > 0)
    # sidecar content present in prefix?
    sc = [t for t in pre if t.role == "system" and "[subagent trajectory:" in (t.content or "")]
    out["sidecar_marker_turns_in_prefix"] = [t.content for t in sc]

# 5. sidecar file shapes
sides = {}
for p in sorted((TRIAL / "agent").glob("trajectory.summarization-*.json")):
    b = p.read_bytes()
    d = json.loads(b)
    sides[p.name] = {"sha256": hashlib.sha256(b).hexdigest()[:16],
                     "n_steps": len(d.get("steps") or []),
                     "agent": (d.get("agent") or {}).get("name"),
                     "summarization_index": ((d.get("agent") or {}).get("extra") or {}).get("summarization_index")}
out["sidecars"] = sides

print(json.dumps(out, indent=2, default=str))
