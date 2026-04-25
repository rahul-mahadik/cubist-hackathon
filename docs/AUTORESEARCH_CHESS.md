# AutoResearch Chess on Cubist

This fork uses Cubist as the orchestration framework and includes a chess-engine target repo under `examples/chess_lab`.

The important shift from the standalone P72 prototype:

- Cubist is now the control plane.
- The chess code is the target repo.
- Cubist's before/after gates control every pod task.
- Cubist's v2 worktrees isolate development tasks.
- Artifacts, budget, events, parent actions, and pod outputs live in the framework-state directory.

## Chess Target

The target is:

```text
examples/chess_lab
```

It includes:

- `engine/candidates/alphabeta`
- `engine/candidates/mcts`
- `engine/candidates/nnue_lite`
- `engine/candidates/policy_guided`
- `tournament/runner.py`
- `data/dev_positions.fen`
- `data/heldout_positions.fen`
- smoke tests for legal moves and eval generation

Run the target tests directly:

```bash
cd examples/chess_lab
python3 -m pytest tests
python3 -m tournament.runner alphabeta --move-budget-ms 200
```

Each tournament run writes a JSON summary and PGN under `artifacts/evals/`.
The JSON includes both `win_rate` and a rough `elo_delta`. The Elo number is
derived from win rate, so treat it as a ranking convenience rather than a
statistically rigorous rating.

Print a compact eval table:

```bash
cd examples/chess_lab
python3 -m tournament.summarize_evals
```

## Local Cubist Run

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Bootstrap a framework-state dir against the chess target:

```bash
.venv/bin/python -m framework --state-dir ./fw-chess run start \
  --goal "Run an AutoResearch experiment comparing alpha-beta, MCTS, NNUE-lite, and policy-guided chess engines under a fixed token budget. Improve candidates through isolated worktrees, run smoke tests and tournament evals, prune weak candidates, and produce a final comparison report." \
  --target-repo "$(pwd)/examples/chess_lab" \
  --overwrite
```

Start the backend:

```bash
.venv/bin/python -m framework --state-dir ./fw-chess backend
```

Start a pod:

```bash
export ANTHROPIC_API_KEY_POD_A=sk-ant-...
.venv/bin/python -m framework --state-dir ./fw-chess start-pod pod_a
```

Optional second pod:

```bash
export ANTHROPIC_API_KEY_POD_B=sk-ant-...
.venv/bin/python -m framework --state-dir ./fw-chess start-pod pod_b
```

Then open Claude Code in the framework-state dir:

```bash
cd fw-chess
claude
```

Ask:

```text
Run a methodology pass for the AutoResearch chess experiment. Create tasks for CandidateCards, cheap evals, candidate improvement in worktrees, pruning, held-out eval, and final report. Use before and after gates for every task.
```

## What To Look At

Framework state:

```bash
.venv/bin/python -m framework --state-dir ./fw-chess state
.venv/bin/python -m framework --state-dir ./fw-chess plan show --include-archived
.venv/bin/python -m framework --state-dir ./fw-chess artifact list
```

Approve every pending before-gate task in one shot when you want real
wall-clock parallelism:

```bash
.venv/bin/python -m framework --state-dir ./fw-chess gate before approve-all
```

Audit files:

```text
fw-chess/framework.db
fw-chess/events.jsonl
fw-chess/logs/parent_actions.jsonl
fw-chess/logs/budget_ledger.jsonl
fw-chess/rolling_summary.md
fw-chess/worktrees/
examples/chess_lab/artifacts/evals/
```

## EC2 Recommendation

For a real hackathon run, use one EC2 instance as the central box.

Recommended:

- Instance: `t3.medium`
- OS: Ubuntu 22.04 or 24.04 LTS
- Disk: 40 GB gp3
- RAM: 4 GB
- Inbound ports:
  - `22` for SSH
  - `8765` for the FastAPI backend if you want remote access
- Install:
  - Python 3.11+
  - git
  - build-essential
  - tmux

`t3.small` can work for a demo, but `t3.medium` is more comfortable because chess evals and multiple pod processes can overlap.

You do not need GPU.

## Keys Needed

Minimal:

```text
ANTHROPIC_API_KEY_POD_A
```

Better for multi-pod:

```text
ANTHROPIC_API_KEY_POD_A
ANTHROPIC_API_KEY_POD_B
```

The parent runs in Claude Code and uses the user's Claude Code login/session, not a pod API key.

## EC2 Setup Commands

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv git build-essential tmux
git clone https://github.com/rahul-mahadik/cubist-hackathon.git
cd cubist-hackathon
git checkout autoresearch-chess-lab
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Use tmux panes:

```bash
tmux new -s chess
```

Pane 1: backend

```bash
.venv/bin/python -m framework --state-dir ./fw-chess backend --host 0.0.0.0 --port 8765
```

Pane 2: pod A

```bash
export ANTHROPIC_API_KEY_POD_A=sk-ant-...
.venv/bin/python -m framework --state-dir ./fw-chess start-pod pod_a
```

Pane 3: pod B

```bash
export ANTHROPIC_API_KEY_POD_B=sk-ant-...
.venv/bin/python -m framework --state-dir ./fw-chess start-pod pod_b
```

Pane 4: parent

```bash
cd fw-chess
claude
```
