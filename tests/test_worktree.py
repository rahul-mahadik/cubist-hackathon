"""Unit tests for framework/worktree.py — git worktree helpers (v2)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from framework.worktree import (
    WorktreeError, auto_commit_all, create_worktree, ensure_branch,
    extract_diff, is_git_repo, merge_into_base, remove_worktree,
)


@pytest.fixture
def repo(tmp_path) -> Path:
    """Tmp git repo with one commit and a configured user."""
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=r, check=True)
    (r / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "."], cwd=r, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=r, check=True)
    return r


def test_is_git_repo_true(repo):
    assert is_git_repo(repo) is True


def test_is_git_repo_false(tmp_path):
    assert is_git_repo(tmp_path / "nope") is False
    plain = tmp_path / "plain"
    plain.mkdir()
    assert is_git_repo(plain) is False


def test_ensure_branch_creates_when_missing(repo):
    ensure_branch(repo, "framework/run-001")
    proc = subprocess.run(
        ["git", "branch", "--list", "framework/run-001"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    assert "framework/run-001" in proc.stdout


def test_ensure_branch_idempotent(repo):
    ensure_branch(repo, "framework/run-001")
    ensure_branch(repo, "framework/run-001")  # no error
    # Still only one branch with that name.
    proc = subprocess.run(
        ["git", "branch", "--list", "framework/run-001"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    assert proc.stdout.count("framework/run-001") == 1


def test_ensure_branch_rejects_non_repo(tmp_path):
    with pytest.raises(WorktreeError):
        ensure_branch(tmp_path / "ghost", "x")


def test_create_worktree_makes_directory_on_branch(repo, tmp_path):
    ensure_branch(repo, "framework/run-001")
    wt_root = tmp_path / "worktrees"
    wt = create_worktree(repo, "framework/run-001", "t_abc", wt_root)
    assert wt.exists() and wt.is_dir()
    assert wt == (wt_root / "t_abc").resolve()
    # The new branch was created at framework/run-001/t_abc.
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=wt, capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "framework/run-001-t_abc"


def test_create_worktree_refuses_existing_path(repo, tmp_path):
    ensure_branch(repo, "framework/run-001")
    wt_root = tmp_path / "worktrees"
    create_worktree(repo, "framework/run-001", "t_abc", wt_root)
    with pytest.raises(WorktreeError, match="already exists"):
        create_worktree(repo, "framework/run-001", "t_abc", wt_root)


def test_two_worktrees_dont_collide(repo, tmp_path):
    """Core v2 use case: two pods working on same repo, separate worktrees."""
    ensure_branch(repo, "framework/run-001")
    wt_root = tmp_path / "worktrees"
    wt_a = create_worktree(repo, "framework/run-001", "t_aaa", wt_root)
    wt_b = create_worktree(repo, "framework/run-001", "t_bbb", wt_root)
    (wt_a / "fizz.py").write_text("a = 1\n")
    (wt_b / "buzz.py").write_text("b = 2\n")
    assert (wt_a / "fizz.py").exists()
    assert not (wt_a / "buzz.py").exists()
    assert (wt_b / "buzz.py").exists()
    assert not (wt_b / "fizz.py").exists()


def test_extract_diff_captures_committed_uncommitted_untracked(repo, tmp_path):
    ensure_branch(repo, "framework/run-001")
    wt = create_worktree(repo, "framework/run-001", "t_xxx", tmp_path / "wt")

    # 1. committed change
    (wt / "a.py").write_text("a = 1\n")
    subprocess.run(["git", "add", "a.py"], cwd=wt, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "-m", "add a"],
        cwd=wt, check=True,
    )
    # 2. uncommitted change (modify the file we just committed)
    (wt / "a.py").write_text("a = 2\n")
    # 3. untracked file
    (wt / "b.py").write_text("b = 3\n")

    diff = extract_diff(wt, "framework/run-001")
    assert "a.py" in diff
    assert "b.py" in diff
    assert "committed:" in diff
    assert "uncommitted:" in diff
    assert "untracked:" in diff


def test_extract_diff_empty_when_no_changes(repo, tmp_path):
    ensure_branch(repo, "framework/run-001")
    wt = create_worktree(repo, "framework/run-001", "t_clean", tmp_path / "wt")
    diff = extract_diff(wt, "framework/run-001")
    assert diff == ""


def test_remove_worktree_cleans_up(repo, tmp_path):
    ensure_branch(repo, "framework/run-001")
    wt = create_worktree(repo, "framework/run-001", "t_rm", tmp_path / "wt")
    (wt / "scratch.py").write_text("x = 1\n")  # uncommitted edit
    remove_worktree(repo, wt)
    assert not wt.exists()


def test_remove_worktree_safe_on_missing(repo, tmp_path):
    """Best-effort cleanup must never raise — gate transitions can't
    block on filesystem hiccups."""
    ghost = tmp_path / "never-existed"
    remove_worktree(repo, ghost)  # no exception


def test_auto_commit_all_picks_up_untracked(repo, tmp_path):
    ensure_branch(repo, "framework/run-001")
    wt = create_worktree(repo, "framework/run-001", "t_x", tmp_path / "wt")
    (wt / "new.py").write_text("x = 1\n")
    made = auto_commit_all(wt, "auto")
    assert made is True
    # Check the commit landed.
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=wt,
        capture_output=True, text=True, check=True,
    )
    assert "auto" in log.stdout


def test_auto_commit_all_noop_when_clean(repo, tmp_path):
    ensure_branch(repo, "framework/run-001")
    wt = create_worktree(repo, "framework/run-001", "t_clean", tmp_path / "wt")
    assert auto_commit_all(wt, "auto") is False


def test_merge_into_base_fast_forwards_first_then_merges_second(repo, tmp_path):
    """The realistic v2 flow: two parallel dev tasks each write a
    different file, both get merged back to base. First merge is FF,
    second creates a merge commit (different files, no conflicts)."""
    ensure_branch(repo, "framework/run-001")

    # Task A: greet.py on framework/run-001-t_a
    wt_a = create_worktree(repo, "framework/run-001", "t_a", tmp_path / "wta")
    (wt_a / "greet.py").write_text("def greet(): pass\n")
    auto_commit_all(wt_a, "[t_a] auto")
    remove_worktree(repo, wt_a)

    # Task B: farewell.py on framework/run-001-t_b
    wt_b = create_worktree(repo, "framework/run-001", "t_b", tmp_path / "wtb")
    (wt_b / "farewell.py").write_text("def farewell(): pass\n")
    auto_commit_all(wt_b, "[t_b] auto")
    remove_worktree(repo, wt_b)

    # Merge A into base (FF), then B (real merge).
    merge_into_base(repo, "framework/run-001", "framework/run-001-t_a")
    merge_into_base(repo, "framework/run-001", "framework/run-001-t_b")

    # Verify base has both files. Use a third worktree to inspect.
    inspect = create_worktree(
        repo, "framework/run-001", "t_inspect", tmp_path / "inspect",
    )
    assert (inspect / "greet.py").exists()
    assert (inspect / "farewell.py").exists()


def test_merge_into_base_raises_on_conflict(repo, tmp_path):
    """Two branches that touch the same file with incompatible changes
    must fail loudly so the gate transition can surface the issue."""
    ensure_branch(repo, "framework/run-001")
    wt_a = create_worktree(repo, "framework/run-001", "t_a", tmp_path / "wta")
    (wt_a / "shared.py").write_text("VAL = 1\n")
    auto_commit_all(wt_a, "[t_a]")
    remove_worktree(repo, wt_a)

    wt_b = create_worktree(repo, "framework/run-001", "t_b", tmp_path / "wtb")
    (wt_b / "shared.py").write_text("VAL = 2\n")
    auto_commit_all(wt_b, "[t_b]")
    remove_worktree(repo, wt_b)

    merge_into_base(repo, "framework/run-001", "framework/run-001-t_a")
    with pytest.raises(WorktreeError, match="merge"):
        merge_into_base(repo, "framework/run-001", "framework/run-001-t_b")
