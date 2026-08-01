# Bootstrap image for prefecthq/prefect, used with:
#   vektori-trace mine --repo prefecthq/prefect \
#     --dockerfile examples/dockerfiles/prefect.Dockerfile \
#     --test-cmd 'python -m pytest -m "not service" -p no:randomly -q' \
#     --language python
#
# Supplying a Dockerfile skips the bootstrap ReAct agent, which makes a mining
# run deterministic and free. It is then on us to say how the suite runs
# (--test-cmd), since nothing inspected the repo.
#
# Chosen over pydantic/airflow on measured numbers (100 merged PRs each,
# post-2025-01, using the miner's own linked-issue filter):
#
#   prefecthq/prefect    20% linked, 19/100 also touch tests
#   pydantic/pydantic    21% linked, 18/100 — but only 748 post-cutoff PRs,
#                        and pydantic-core is a Rust extension whose wheel must
#                        match every base commit
#   apache/airflow        2% linked — genuinely does not link issues
#
# prefect has 3,735 merged PRs since 2025-01-01, which is the only pool on the
# shortlist large enough to reach N≈150 from one repo.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# prefect declares its test deps in PEP 735 `[dependency-groups]`, not
# `[project.optional-dependencies]` — the same trap documented in
# structlog.Dockerfile. `pip install -e '.[dev]'` resolves to nothing and leaves
# pytest absent, so the suite "fails" for every PR and every task skips as
# no_fail_to_pass. uv understands dependency groups; pip does not.
RUN pip install --no-cache-dir uv

# `--inexact` keeps the editable install from being pruned on later syncs.
# `.[otel]` is not optional in practice: tests/conftest.py imports
# `tests.fixtures.telemetry` unconditionally, which imports
# `opentelemetry.sdk`. Without it *every* collection fails, the suite is red at
# both base and patched commit, and every PR skips as `no_fail_to_pass` — the
# same silent-zero failure mode structlog's `[tests]` group had.
#
# watchfiles / whenever / yamllint are imported by 9 more test modules
# (tests/cli/test_dev.py, tests/deployment/test_yamllint.py,
# tests/server/schemas/test_schedules.py, …). Missing, they drop out of the
# collected set, which would quietly shrink every P2P regression guard.
RUN uv pip install --system --no-cache -e ".[otel]" \
    && uv pip install --system --no-cache \
        pytest pytest-asyncio pytest-timeout pytest-env pytest-xdist \
        moto numpy jinja2 pluggy respx opentelemetry-test-utils \
        watchfiles whenever yamllint

# prefect's own pytest config: `testpaths = ["tests"]` (so the
# `src/integrations/*` subpackages are out of scope by default — they need
# their own installs and would drift independently), `asyncio_mode = "auto"`,
# and a `service(arg)` marker for tests needing Docker/external services.
# `-m "not service"` deselects those; they cannot run inside the mining
# sandbox and would otherwise land in P2P as permanent failures.
#
# NOTE: prefect sets `filterwarnings = ["error"]`. A deprecation introduced by
# a dependency released *after* a given base commit turns that commit's suite
# red for reasons unrelated to the PR, which shows up as `no_fail_to_pass`. If
# yield comes in low, this is the first thing to check — and the argument for
# keeping the mining window recent (see below).
ENV PREFECT_HOME=/tmp/prefect
ENV PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:////tmp/prefect/prefect.db"

# `-e` above means a `git reset --hard <base_commit>` swaps the source under the
# installed package with no reinstall needed — provided the dependency set has
# not moved. Over a 19-month window it will have; the miner's dependency-drift
# detector catches this and skips affected PRs. Mining in ~2-quarter slices with
# a bootstrap image per slice keeps drift-skips down.
RUN git config --global --add safe.directory /workspace
