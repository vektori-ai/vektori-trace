"""Command-line interface.

`cli.py` was one 2,965-line module: 22 command bodies, their private helpers,
an 843-line `build_parser`, and `main`. It is now a package — one module per
command group under `commands/`, the argparse tree in `parser`, the entry
point in `main`.

Everything the old module exposed is re-exported here, so `vektori_trace.cli:main`
(the console script) and `from vektori_trace.cli import build_parser, cmd_route`
keep working unchanged.

One caveat for tests: patching `vektori_trace.cli.<helper>` now patches this
re-export, not the name the command body actually reads. Patch the defining
module instead — e.g. `vektori_trace.cli._shared._load_traces`.
"""

# Private, but the test suite imports several of these from `vektori_trace.cli`
# directly, so the old module's full surface is kept here on purpose.
from ._args import (  # noqa: F401
    _add_endpoint_args,
    _min_gap_arg,
    _min_support_arg,
    _model_info_arg,
    _positive_int_arg,
)
from ._shared import _check_replay_models, _load_traces  # noqa: F401
from .commands.capture import cmd_capture_proxy
from .commands.diagnose import cmd_diagnose
from .commands.distill import cmd_distill
from .commands.env import cmd_checkenv, cmd_prove, cmd_selftest
from .commands.evaluate import cmd_import_gym, cmd_passk
from .commands.ground import cmd_ground
from .commands.mine import cmd_mine, cmd_mine_commits
from .commands.replay import cmd_replay
from .commands.resume import cmd_bisect, cmd_resume_check
from .commands.route import (  # noqa: F401
    _reload_routing_decisions,
    cmd_plan_b_arms,
    cmd_route,
)
from .commands.select import cmd_select
from .commands.teacher import (
    cmd_align_report,
    cmd_build_bridge,
    cmd_check_tokenizers,
    cmd_probe_teacher,
)
from .commands.train import cmd_run_arms, cmd_train
from .main import main
from .parser import build_parser

__all__ = [
    "build_parser",
    "cmd_align_report",
    "cmd_bisect",
    "cmd_build_bridge",
    "cmd_capture_proxy",
    "cmd_check_tokenizers",
    "cmd_checkenv",
    "cmd_diagnose",
    "cmd_distill",
    "cmd_ground",
    "cmd_import_gym",
    "cmd_mine",
    "cmd_mine_commits",
    "cmd_passk",
    "cmd_plan_b_arms",
    "cmd_probe_teacher",
    "cmd_prove",
    "cmd_replay",
    "cmd_resume_check",
    "cmd_route",
    "cmd_run_arms",
    "cmd_select",
    "cmd_selftest",
    "cmd_train",
    "main",
]
