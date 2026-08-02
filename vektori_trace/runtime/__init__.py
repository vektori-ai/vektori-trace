"""Where models and tasks actually execute.

`modal_env` and `serve` bring a vLLM endpoint up; `endpoint` is the client
that talks to one. `envcheck` builds and probes the sandboxed task
environments, and `rollout` drives a policy through a task to produce traces.

No re-exports — import the submodule you need.

One edge runs the other way on purpose: `serve` imports `train` lazily, inside
the function that stages a local adapter onto a Modal volume, because `train`
imports `runtime.modal_env` at module scope. Keep that import deferred.
"""
