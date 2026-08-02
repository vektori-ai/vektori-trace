"""One module per command group.

Each module holds the `cmd_*` bodies and the helpers only that group uses.
Anything shared across groups lives in `cli._shared`; argparse `type=`
validators live in `cli._args`. Nothing here imports `cli.parser` — the
dependency runs parser -> commands, never back.
"""
