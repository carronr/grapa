import os
import shutil
import stat
import subprocess
import sys
import tarfile
import uuid
import zipfile
from pathlib import Path

import pytest


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _run_checked(cmd, cwd, env):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _ignore_build_copy(_dir, names):
    ignored = {
        ".git",
        ".github",
        ".idea",
        ".pytest_cache",
        ".pytest-build_sdist",
        ".tmp-build",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "grapa.egg-info",
    }
    return [name for name in names if name in ignored]


def _copy_writable(src, dst):
    shutil.copyfile(src, dst)
    os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)
    return dst


def _assert_sdist_contains_package_files(sdist_path: Path):
    with tarfile.open(sdist_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert any(name.endswith("grapa/__init__.py") for name in names)
    assert any(name.endswith("grapa/shared/keywordsdata_curve.txt") for name in names)
    assert any(name.endswith("grapa/frontend/datareading.ico") for name in names)


def _assert_wheel_contains_package_files(wheel_path: Path):
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
    assert "grapa/__init__.py" in names
    assert "grapa/shared/keywordsdata_curve.txt" in names
    assert "grapa/frontend/datareading.ico" in names


@pytest.mark.long
def test_package_build_and_twine_check():
    """Build package artifacts and verify them with `twine check`."""
    repo_root = _repo_root()
    tmp_root = repo_root / ".pytest-build_sdist" / f"package-build-{uuid.uuid4().hex}"
    source_dir = tmp_root / "source"
    dist_dir = tmp_root / "dist"
    build_tmp = tmp_root / "build-tmp"
    tmp_root.mkdir(parents=True)
    shutil.copytree(
        repo_root,
        source_dir,
        ignore=_ignore_build_copy,
        copy_function=_copy_writable,
    )
    dist_dir.mkdir()
    build_tmp.mkdir()

    env = os.environ.copy()
    env["TMP"] = str(build_tmp)
    env["TEMP"] = str(build_tmp)

    build_cmd = [
        sys.executable,
        "-m",
        "build",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    try:
        try:
            _run_checked(build_cmd, cwd=source_dir, env=env)
        except subprocess.CalledProcessError as exc:
            output = f"{exc.stdout}\n{exc.stderr}"
            if "PermissionError" not in output and "Zugriff verweigert" not in output:
                raise
            fallback_sdist_cmd = [
                sys.executable,
                "setup.py",
                "sdist",
                "--dist-dir",
                str(dist_dir),
            ]
            fallback_wheel_cmd = [
                sys.executable,
                "setup.py",
                "bdist_wheel",
                "--dist-dir",
                str(dist_dir),
            ]
            _run_checked(fallback_sdist_cmd, cwd=source_dir, env=env)
            _run_checked(fallback_wheel_cmd, cwd=source_dir, env=env)
        artifacts = sorted(dist_dir.iterdir())
        assert len(artifacts) >= 2
        wheel_path = next(path for path in artifacts if path.suffix == ".whl")
        sdist_path = next(path for path in artifacts if path.name.endswith(".tar.gz"))
        _assert_wheel_contains_package_files(wheel_path)
        _assert_sdist_contains_package_files(sdist_path)
        check_cmd = [sys.executable, "-m", "twine", "check", *map(str, artifacts)]
        _run_checked(check_cmd, cwd=source_dir, env=env)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
