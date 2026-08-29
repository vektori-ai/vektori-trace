#!/usr/bin/env python3
"""No-spend preflight: is this box ready to run the pilot?

Every check is local -- no GPU, no endpoint, no teacher call, no network
beyond what git already fetched. Run it with the venv interpreter
(`.venv/bin/python`); the box's bare python3 is 3.10 and cannot import the
package at all (`StrEnum` needs 3.11+), which is itself worth catching here
rather than at the first paid step.

    .venv/bin/python scripts/tau2_pilot_preflight.py [--repo /data/vektori-trace]
"""

import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--repo", default="/data/vektori-trace")
_ap.add_argument("--expect-commit", default="7d43f9de3d481220e449265b1b13f0df75243ee6")
_ap.add_argument("--expect-manifest-sha", default="f00004d4d3f0dd1a")
_ap.add_argument("--run-id", default="pilot_10x8_20260829b")
_A = _ap.parse_args()
R = Path(_A.repo)
ok = True
def chk(name, good, detail=""):
    global ok
    ok = ok and good
    print(f"[{'PASS' if good else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))

sha = subprocess.run(["git","-c","safe.directory=/data/vektori-trace","rev-parse","HEAD"],
                     cwd=R, capture_output=True, text=True).stdout.strip()
chk("git commit", sha == _A.expect_commit, sha)
# Only MODIFIED tracked files matter. Untracked scratch (.bak, .box-old)
# cannot change what the code does, and failing on them would make the
# preflight cry wolf on a box people actually work on.
dirty = [l for l in subprocess.run(
    ["git","-c",f"safe.directory={R}","status","--porcelain",
     "vektori_trace","scripts","docs/prereg"],
    cwd=R, capture_output=True, text=True).stdout.splitlines()
    if not l.startswith("??")]
chk("no modified tracked source", not dirty, "; ".join(dirty)[:120] or "clean")
chk("interpreter >= 3.11", sys.version_info >= (3, 11),
    f"{sys.version_info.major}.{sys.version_info.minor}")

mp = R/f"docs/prereg/{_A.run_id}.manifest.json"
chk("manifest present", mp.exists(), str(mp))
if mp.exists():
    mh = hashlib.sha256(mp.read_bytes()).hexdigest()[:16]
    chk("manifest sha256:16", mh == _A.expect_manifest_sha, mh)
    m = json.loads(mp.read_text())
    chk("plan_hash", m["plan_hash"] == "aa9251ccb6d566fa", m["plan_hash"])
    chk("parent adapter", m["adapter_hash"] == "3869b147ab7ce5d2", m["adapter_hash"])
    flat = [(p["task_id"],p["seed"]) for b in m["plans_by_update"] for p in b]
    chk("80 distinct pairs", len(flat)==80==len(set(flat)), str(len(flat)))
    chk("parser/proj/algo recorded",
        (m.get("parser_version"),m.get("projection_version"),m.get("score_algorithm"))
        ==("v2","v1","chunk-v2"),
        f"{m.get('parser_version')}/{m.get('projection_version')}/{m.get('score_algorithm')}")
    chk("thinking_mode", m.get("thinking_mode")=="thinking", str(m.get("thinking_mode")))
    chk("supersedes failed run",
        (m.get("supersedes") or {}).get("run_id")=="pilot_10x8_20260829")

sys.path.insert(0, str(R))
from vektori_trace.tau2.live_agent import PARSER_VERSION, split_generation
from vektori_trace.tau2.live_projection import PROJECTION_VERSION
from vektori_trace.tau2.live_score import SCORE_ALGORITHM
chk("code versions match manifest",
    (PARSER_VERSION,PROJECTION_VERSION,SCORE_ALGORITHM)==("v2","v1","chunk-v2"))

raw='<think>need order\n<tool_call>{"name":"get_order_details","arguments":{"order_id":"W1"}}</tool_call>'
r,c,t = split_generation(raw)
chk("unclosed-think parses", bool(r and r.strip()) and len(t)==1, repr((r or "")[:20]))

from vektori_trace.tau2.live_batch import projected_turn_advantages
from vektori_trace.tau2.live_score import ProjectedChunk
ta = projected_turn_advantages(turn_index=0, action_token_ids=[1,2,3],
        behavior_logprobs=[-0.5,-1.0,-1.5],
        chunks=[ProjectedChunk("reasoning:0","reasoning",(0,1,2),(-3.0,))])
chk("N:1 chunk rule -> zeros", all(abs(a)<1e-12 for a in ta.advantages), str(ta.advantages))

def reusable(rows, by_key):
    seen={}
    for r_ in rows:
        if r_.get("key") and r_.get("projection")=="semantic": seen[r_["key"]]=r_
    keep={}
    for k,r_ in seen.items():
        if r_.get("score_algorithm")!=SCORE_ALGORITHM or "chunks" not in r_: continue
        w=(by_key.get(k) or {}).get("score_fingerprint")
        if w is not None and r_.get("fingerprint")!=w: continue
        keep[k]=r_
    return keep
chk("flat cache rejected",
    reusable([{"key":"a","projection":"semantic","teacher_logprob_by_index":{"0":-1.0}}],{})=={})
chk("fingerprint mismatch rejected",
    reusable([{"key":"a","projection":"semantic","score_algorithm":SCORE_ALGORITHM,
               "chunks":[],"fingerprint":"old"}],{"a":{"score_fingerprint":"new"}})=={})
chk("valid cache reused",
    set(reusable([{"key":"a","projection":"semantic","score_algorithm":SCORE_ALGORITHM,
               "chunks":[],"fingerprint":"fp"}],{"a":{"score_fingerprint":"fp"}}))=={"a"})

from vektori_trace.runtime.serve import resolve_web_url
U="https://ws--vektori-trace-serve-vllmserver-openai-compat-dev.modal.run"
class G:
    def get_web_url(self): return U
chk("URL resolves", resolve_web_url(G())==U)
try:
    resolve_web_url(); chk("URL never fabricated", False, "did not raise")
except RuntimeError as e:
    chk("URL never fabricated", "modal.run" not in str(e))

# Presence only -- the value is never printed, compared or logged. A key
# that is merely NAMED in a file is not enough: an empty assignment would
# pass a substring check and then fail at the first paid call.
def _fireworks_source() -> str:
    v = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if v:
        return f"env:FIREWORKS_API_KEY (len {len(v)})"
    for f in (R/".env", R/".env.run", R/".env.rollout",
              Path.home()/".fireworks/auth.ini"):
        try:
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                line = line.strip().lstrip("export ").strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, val = line.partition("=")
                if "FIREWORKS" in k.upper() and "KEY" in k.upper():
                    val = val.strip().strip("'\"")
                    if val:
                        return f"{f.name}:{k.strip()} (len {len(val)})"
        except OSError:
            continue
    return ""

fw = _fireworks_source()
chk("Fireworks secret present", bool(fw), fw or "not found in env, .env, "
    ".env.run, .env.rollout or ~/.fireworks/auth.ini")

app = subprocess.run("modal app list 2>/dev/null | grep -c ephemeral", shell=True,
                     capture_output=True, text=True, cwd=R).stdout.strip() or "0"
chk("0 ephemeral modal apps", app in ("0",""), app)
gp = subprocess.run("nvidia-smi -L 2>/dev/null | wc -l", shell=True,
                    capture_output=True, text=True).stdout.strip()
chk("no local GPUs in use", gp in ("0",""), gp)
tm = subprocess.run("tmux ls 2>&1 | head -1", shell=True, capture_output=True, text=True).stdout.strip()
chk("no tmux sessions", "no server running" in tm or tm=="", tm[:60])

print("\nPREFLIGHT " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
