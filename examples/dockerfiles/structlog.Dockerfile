# Bootstrap image for hynek/structlog, used with:
#   vektori-trace mine --repo hynek/structlog \
#     --dockerfile examples/dockerfiles/structlog.Dockerfile \
#     --test-cmd 'python -m pytest -p no:randomly -q' --language python
#
# Supplying a Dockerfile skips the bootstrap ReAct agent, which makes a mining
# run deterministic and free. It is then on us to say how the suite runs
# (--test-cmd), since nothing inspected the repo.
#
# structlog is pure Python with no compiled deps, so one image builds every base
# commit in recent history — the property that makes a repo cheap to mine.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# structlog declares its test deps in PEP 735 `[dependency-groups]`, not
# `[project.optional-dependencies]`, so `pip install -e '.[tests]'` resolves to
# nothing and leaves pytest absent — the suite then "fails" for every PR and
# every task skips as no_fail_to_pass. Named explicitly instead.
#
# twisted and rich are optional integrations; without them their test modules
# fail at import and drop out of the collected set, which would quietly shrink
# every P2P regression guard.
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir \
        'pytest>=6.0' pytest-asyncio pytest-randomly simplejson 'time-machine>=2.14.1' \
        twisted rich

# `-e` above means a `git reset --hard <base_commit>` swaps the source under the
# installed package with no reinstall needed.
RUN git config --global --add safe.directory /workspace
