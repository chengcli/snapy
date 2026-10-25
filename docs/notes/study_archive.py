"""Verify or restore the archived study data without overwriting changed files."""

import argparse
import hashlib
import tarfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "cubed_sphere_study_data.tar.gz"
CHECKSUMS = "ARCHIVE_SHA256SUMS"


def verified_files(archive=ARCHIVE):
    """Return regular-file contents only after validating every member and hash."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive members")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != member.name
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
        contents = {m.name: tar.extractfile(m).read() for m in members}
    checksums = contents.pop(CHECKSUMS).decode("utf-8").splitlines()
    expected = {}
    for line in checksums:
        digest, name = line.split("  ", 1)
        if name in expected:
            raise ValueError(f"Duplicate checksum: {name}")
        expected[name] = digest
    if contents.keys() != expected.keys():
        raise ValueError("Archive contents do not match the checksum inventory")
    for name, data in contents.items():
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError(f"Checksum mismatch: {name}")
    return contents


def read_data(path):
    """Read a loose file, or its verified archived copy under docs/notes."""
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    relative = path.resolve().relative_to(HERE).as_posix()
    try:
        return verified_files()[relative]
    except (FileNotFoundError, KeyError) as error:
        raise FileNotFoundError(f"No loose or archived study file: {path}") from error


def restore(contents, directory):
    """Preflight all paths; retain identical files and refuse conflicting copies."""
    root = Path(directory).resolve()
    pending = []
    for name, data in contents.items():
        target = root / name
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError(f"Refusing a symlink destination: {target}")
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise FileExistsError(f"Refusing to overwrite changed file: {target}")
        else:
            pending.append((target, data))
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "extract"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--directory", type=Path, default=HERE)
    args = parser.parse_args()
    contents = verified_files(args.archive)
    print(f"Verified {len(contents)} files, {sum(map(len, contents.values())):,} bytes (SHA-256)")
    if args.action == "extract":
        count = restore(contents, args.directory)
        print(f"Restored {count} files to {args.directory.resolve()}; identical files retained")


if __name__ == "__main__":
    main()
