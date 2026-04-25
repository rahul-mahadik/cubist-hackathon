"""Git worktree helpers for parallel candidate development (v2 / Phase 7).

The framework creates a per-task worktree at ``<state_dir>/worktrees/<task_id>/``
when a development task is approved at the before gate. The pod's tools
are sandboxed to that worktree, so two pods can edit the same logical
repo without stepping on each other.

Branch naming: ``<base_branch>-<task_id>``. Note the dash, not slash —
git refs are filesystem paths under ``.git/refs/heads/``, so a slash
between base and task makes git try to nest the task ref *under* the
base ref's path, which fails because the base is already a file at
that location ("cannot create 'refs/heads/X/Y': 'refs/heads/X' exists").
Dashes keep the task branches as siblings of the base branch under
``.git/refs/heads/<prefix>/``. The base branch (typically
``framework/<run-id>``) is created by ``framework run start``. The
worktree branches off the base, accumulates the pod's edits as commits
the model can ``git commit``, and is merged or cherry-picked manually
by the user after after-gate approval.

We deliberately keep this module thin — three functions, all subprocess
wrappers around ``git``. No git library, no fancy state. The caller
(services.py) handles the SQL side.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeError(RuntimeError):
    """git worktree operation failed (non-zero exit)."""


def _git(args: list[str], *, cwd: str | Path | None = None,
         check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True, text=True,
        timeout=60,
    )
    if check and proc.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed (cwd={cwd}, exit={proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def is_git_repo(path: str | Path) -> bool:
    """Return True if ``path`` is the working tree of a git repo."""
    p = Path(path)
    if not p.is_dir():
        return False
    proc = _git(["rev-parse", "--is-inside-work-tree"], cwd=p, check=False)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def ensure_branch(target_repo: str | Path, branch: str) -> None:
    """Create ``branch`` off current HEAD in ``target_repo`` if it doesn't exist.

    No-op when the branch already exists. Used by ``framework run start``
    so per-task worktrees have a stable base to fork from.
    """
    target = Path(target_repo)
    if not is_git_repo(target):
        raise WorktreeError(f"{target} is not a git repository")
    # `show-ref --verify --quiet refs/heads/<branch>` exits 0 if the branch exists.
    exists = _git(
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=target, check=False,
    ).returncode == 0
    if not exists:
        _git(["branch", branch], cwd=target)


def create_worktree(
    target_repo: str | Path,
    base_branch: str,
    task_id: str,
    worktrees_root: str | Path,
) -> Path:
    """Create a worktree for ``task_id`` and return its absolute path.

    The worktree branches off ``base_branch`` into a new branch named
    ``<base_branch>/<task_id>``. The branch is created on the worktree's
    initial checkout — we don't try to detect "already exists" because
    that would mean a task is being claimed twice (a bug elsewhere).
    """
    target = Path(target_repo).resolve()
    if not is_git_repo(target):
        raise WorktreeError(f"{target} is not a git repository")
    wt_root = Path(worktrees_root)
    wt_root.mkdir(parents=True, exist_ok=True)
    worktree_path = (wt_root / task_id).resolve()
    if worktree_path.exists():
        raise WorktreeError(
            f"worktree path {worktree_path} already exists; "
            "did the task get approved twice without cleanup?"
        )
    branch_name = f"{base_branch}-{task_id}"
    _git(
        ["worktree", "add", "-b", branch_name, str(worktree_path), base_branch],
        cwd=target,
    )
    return worktree_path


def extract_diff(worktree_path: str | Path, base_branch: str) -> str:
    """Return ``git diff <base_branch>...HEAD`` from inside the worktree.

    Captures both committed and uncommitted changes the pod made. Empty
    string means no changes — the model produced an artifact but didn't
    edit anything (which is suspicious but not the framework's problem).
    """
    wt = Path(worktree_path)
    if not wt.exists():
        return ""
    # Three-dot diff: changes on the worktree branch since it diverged
    # from base, *plus* any uncommitted changes in the working tree.
    committed = _git(
        ["diff", f"{base_branch}...HEAD"], cwd=wt, check=False,
    ).stdout
    uncommitted = _git(
        ["diff"], cwd=wt, check=False,
    ).stdout
    untracked = _git(
        ["ls-files", "--others", "--exclude-standard"],
        cwd=wt, check=False,
    ).stdout
    parts = []
    if committed:
        parts.append("# committed:\n" + committed)
    if uncommitted:
        parts.append("# uncommitted:\n" + uncommitted)
    if untracked.strip():
        parts.append("# untracked:\n" + untracked)
    return "\n".join(parts)


def auto_commit_all(
    worktree_path: str | Path, message: str,
) -> bool:
    """Stage and commit everything in the worktree. Returns True if a
    commit was made, False if there was nothing to commit.

    Pods don't always remember to ``git commit`` — they edit files via
    the write_file tool and trust the framework to capture the result.
    Without this auto-commit, ``git worktree remove`` would silently
    drop uncommitted changes when the task is cleaned up.
    """
    wt = Path(worktree_path)
    if not wt.exists():
        return False
    # Configure user only for this commit so the worktree doesn't depend
    # on the user's git config (which may not be set on a CI machine).
    _git(["add", "-A"], cwd=wt, check=False)
    status = _git(["status", "--porcelain"], cwd=wt, check=False).stdout
    if not status.strip():
        return False
    _git(
        ["-c", "user.email=framework@local",
         "-c", "user.name=framework",
         "commit", "-m", message],
        cwd=wt,
    )
    return True


def merge_into_base(
    target_repo: str | Path, base_branch: str, task_branch: str,
) -> None:
    """Merge ``task_branch`` into ``base_branch`` on the target repo.

    Uses a temporary worktree so the user's main checkout (typically on
    ``main`` or whatever they had before bootstrap) isn't disturbed.
    Default merge strategy — sibling task branches don't touch the same
    files (by methodology, each dev task scopes a single change), so
    conflicts shouldn't happen. If they do, we abort the merge and
    raise so the gate transition fails loudly rather than leaving a
    half-merged base.

    Concurrency note: parallel after-gate approvals could race here.
    For v2 we rely on the gate being user-driven (sequential), but a
    future hardening would file-lock or queue these.
    """
    import tempfile
    target = Path(target_repo).resolve()
    if not is_git_repo(target):
        raise WorktreeError(f"{target} is not a git repository")
    with tempfile.TemporaryDirectory(prefix="fw-merge-") as td:
        merge_wt = Path(td) / "merge"
        _git(["worktree", "add", str(merge_wt), base_branch], cwd=target)
        try:
            proc = _git(
                ["-c", "user.email=framework@local",
                 "-c", "user.name=framework",
                 "merge", "--no-edit", task_branch],
                cwd=merge_wt, check=False,
            )
            if proc.returncode != 0:
                _git(["merge", "--abort"], cwd=merge_wt, check=False)
                raise WorktreeError(
                    f"merge of {task_branch!r} into {base_branch!r} failed:\n"
                    f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
                )
        finally:
            _git(
                ["worktree", "remove", "--force", str(merge_wt)],
                cwd=target, check=False,
            )


def remove_worktree(target_repo: str | Path, worktree_path: str | Path) -> None:
    """Tear down a worktree. Best-effort — never raises.

    Uses ``--force`` so uncommitted edits don't block cleanup; the diff
    has already been captured by ``extract_diff`` if the caller wanted
    to keep a record. The branch the worktree was on is left in place
    (the user may want to merge / inspect it later).
    """
    target = Path(target_repo)
    wt = Path(worktree_path)
    if not wt.exists():
        return
    _git(
        ["worktree", "remove", "--force", str(wt)],
        cwd=target, check=False,
    )
    # Defensive: if `git worktree remove` failed but the path is stale
    # metadata, force-clean the directory so the next claim doesn't trip
    # the "already exists" check in create_worktree.
    if wt.exists():
        import shutil
        shutil.rmtree(wt, ignore_errors=True)
