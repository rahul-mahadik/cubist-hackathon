"""Phase 3: framework run start bootstrap."""
import subprocess

import pytest
import yaml

from framework.bootstrap import bootstrap_run
from framework.state import StatePaths


def test_bootstrap_creates_full_layout(tmp_path):
    target = tmp_path / "fake-repo"
    target.mkdir()
    state = tmp_path / "fw"

    info = bootstrap_run(state, goal="Implement UCI", target_repo=str(target))

    paths = StatePaths(state)
    assert paths.db.exists()
    assert paths.config_yaml.exists()
    assert paths.parent_claude_md.exists()
    assert (paths.agents_dir / "methodology.md").exists()
    assert (paths.agents_dir / "development.md").exists()
    assert (paths.agents_dir / "testing.md").exists()
    assert paths.rolling_summary.exists()
    assert paths.progress_md.exists()
    assert (paths.root / "run.yaml").exists()

    # Parent CLAUDE.md mentions both gates
    md = paths.parent_claude_md.read_text()
    assert "before gate" in md.lower()
    assert "after gate" in md.lower()
    assert "rolling_summary" in md.lower()

    # config.yaml round-trips
    cfg = yaml.safe_load(paths.config_yaml.read_text())
    assert cfg["models"]["sonnet"] == "claude-sonnet-4-6"

    # run.yaml records the goal + target_repo + branch_name
    run = yaml.safe_load((paths.root / "run.yaml").read_text())
    assert run["goal"] == "Implement UCI"
    assert run["target_repo"] == str(target.resolve())
    assert run["branch_name"].startswith("framework/")

    # Rolling summary contains the goal
    assert "Implement UCI" in paths.rolling_summary.read_text()

    assert info["run_id"] == run["run_id"]
    assert "agents/methodology.md" in info["files_created"]


def test_bootstrap_refuses_existing_dir_without_overwrite(tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    state = tmp_path / "fw"
    state.mkdir()
    (state / "stale_file").write_text("x")
    with pytest.raises(FileExistsError):
        bootstrap_run(state, goal="g", target_repo=str(target))


def test_bootstrap_overwrite_wipes_old(tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    state = tmp_path / "fw"
    state.mkdir()
    (state / "stale").write_text("x")
    bootstrap_run(state, goal="g", target_repo=str(target), overwrite=True)
    assert not (state / "stale").exists()


def test_bootstrap_validates_target_repo(tmp_path):
    with pytest.raises(FileNotFoundError):
        bootstrap_run(tmp_path / "fw", goal="g", target_repo=str(tmp_path / "nope"))


def test_bootstrap_creates_framework_branch_in_git_target(tmp_path):
    """v2: when the target is a git repo, bootstrap creates the
    framework/<run-id> branch so per-task worktrees can fork from it."""
    target = tmp_path / "git-repo"
    target.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=target, check=True)
    (target / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=target, check=True)

    state = tmp_path / "fw"
    info = bootstrap_run(state, goal="g", target_repo=str(target))
    assert info["branch_name"].startswith("framework/")

    branch = info["branch_name"]
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
        cwd=target, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"branch {branch!r} should exist after bootstrap; "
        f"git said: {proc.stderr}"
    )

    run = yaml.safe_load((state / "run.yaml").read_text())
    assert run["target_is_git"] is True


def test_bootstrap_tolerates_non_git_target(tmp_path):
    """A plain directory still bootstraps fine — pods just won't get
    worktree isolation."""
    target = tmp_path / "plain"
    target.mkdir()
    state = tmp_path / "fw"
    bootstrap_run(state, goal="g", target_repo=str(target))
    run = yaml.safe_load((state / "run.yaml").read_text())
    # Plain dir → target_is_git: False, no branch created.
    assert run["target_is_git"] is False


def test_bootstrap_seeds_gitignore_on_framework_branch(tmp_path):
    """The auto-commit step on after-gate runs `git add -A` in each
    per-task worktree. Without a .gitignore on the framework branch,
    __pycache__ and friends end up committed alongside source.
    Bootstrap should seed a default .gitignore as the first commit on
    the framework branch (only when the user's branch HEAD doesn't
    already track one)."""
    target = tmp_path / "git-repo"
    target.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=target, check=True)
    (target / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=target, check=True)

    state = tmp_path / "fw"
    info = bootstrap_run(state, goal="g", target_repo=str(target))
    branch = info["branch_name"]

    # The framework branch tracks .gitignore now.
    proc = subprocess.run(
        ["git", "cat-file", "-p", f"{branch}:.gitignore"],
        cwd=target, capture_output=True, text=True,
    )
    assert proc.returncode == 0, "framework branch should have .gitignore"
    assert "__pycache__" in proc.stdout
    assert ".pytest_cache" in proc.stdout

    # The user's main branch is untouched.
    proc = subprocess.run(
        ["git", "cat-file", "-e", "main:.gitignore"],
        cwd=target, capture_output=True,
    )
    assert proc.returncode != 0, "main branch must not be touched"


def test_bootstrap_does_not_overwrite_existing_gitignore(tmp_path):
    """If the user's HEAD already tracks .gitignore, the framework
    branch inherits theirs — we don't clobber it with the default."""
    target = tmp_path / "git-repo"
    target.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=target, check=True)
    (target / ".gitignore").write_text("# user's choice\nbuild/\n")
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=target, check=True)

    info = bootstrap_run(tmp_path / "fw", goal="g", target_repo=str(target))
    branch = info["branch_name"]

    proc = subprocess.run(
        ["git", "cat-file", "-p", f"{branch}:.gitignore"],
        cwd=target, capture_output=True, text=True, check=True,
    )
    assert "user's choice" in proc.stdout
    assert "build/" in proc.stdout
    # Default entries should NOT have been added.
    assert "__pycache__" not in proc.stdout
