"""Build a reproducible Blender extension archive using the Python standard library.

Ordinary Python 3.11+:  python scripts/build_extension.py [--output path.zip]
Blender Python:        blender --background --factory-startup --python
                       scripts/build_extension.py -- [--output path.zip]

Run Blender's official validator separately on the resulting archive:
    blender --background --factory-startup --command extension validate path.zip
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "smart_lod"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
EXCLUDED_DIRECTORIES = {"__pycache__", "tests", ".git", ".pytest_cache"}
# Modules the add-on imports at runtime; a build without any of them is refused.
REQUIRED_MODULES = ("__init__.py", "geometry.py", "metrics.py", "perception.py", "pipeline.py",
                    "qem.py", "raster.py", "ui.py")


def archive_members() -> dict[str, Path]:
    """Include only extension Python source, manifest, license and user guide."""
    members = {"blender_manifest.toml": SOURCE / "blender_manifest.toml"}
    for path in SOURCE.rglob("*.py"):
        relative = path.relative_to(SOURCE)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        members[relative.as_posix()] = path
    members["LICENSE"] = ROOT / "LICENSE"
    members["README.md"] = ROOT / "README.md"
    missing = [name for name in REQUIRED_MODULES if name not in members]
    if missing:
        raise FileNotFoundError(f"Missing extension module(s) in {SOURCE}: {', '.join(missing)}")
    for name, path in members.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing archive input {name}: {path}")
    return members


def build(output: Path | None = None) -> Path:
    members = archive_members()
    with members["blender_manifest.toml"].open("rb") as stream:
        manifest = tomllib.load(stream)
    default_name = f"{manifest['id']}-{manifest['version']}.zip"
    destination = (output or (ROOT / "dist" / default_name)).expanduser().resolve()
    if destination.suffix.lower() != ".zip":
        raise ValueError("Output filename must end in .zip")
    if destination in {path.resolve() for path in members.values()}:
        raise ValueError("Output must not overwrite an archive source file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".smart_lod_", suffix=".tmp",
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for name, path in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            damaged = archive.testzip()
            if damaged is not None:
                raise ValueError(f"Archive CRC validation failed: {damaged}")
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"Built {destination}")
    print(f"Files: {len(members)} | Bytes: {destination.stat().st_size}")
    print(f"SHA256: {digest}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destination .zip file (default: dist/id-version.zip)")
    arguments = parser.parse_args(argv)
    try:
        build(arguments.output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Build failed: {exc}\n")
    return 0


if __name__ == "__main__":
    # Blender retains its own flags in sys.argv; script arguments follow '--'.
    if "--" in sys.argv:
        script_args = sys.argv[sys.argv.index("--") + 1:]
    elif "--python" in sys.argv or "-P" in sys.argv:
        script_args = []
    else:
        script_args = sys.argv[1:]
    raise SystemExit(main(script_args))
