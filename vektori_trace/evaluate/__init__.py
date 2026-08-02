"""Measuring whether a task or a policy actually works.

`validity` is the primitive the rest sit on: run a trial, see if the reward
fires. `passk` and `passrate` sample it; `nonregression` and `planted` are the
guards that catch a task that passes for the wrong reason. `diagnose` turns
traces into labelled deficits, `report` renders them, and `intervene` builds
the counterfactual replays that `reopd` trains on.

Deliberately no re-exports — import the submodule you need. Callers outside
this package (`arms`, `rollout`, `taskgen`, `select`, `reopd`, `mining.miner`,
and the CLI) depend inward on these; nothing here imports back out beyond the
leaves `schema`, `llm`, and `resume`, which keeps the package acyclic.
"""
