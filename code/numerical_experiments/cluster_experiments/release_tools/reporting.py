"""Archive publication/report sources under a new, nonoverwritable review label."""

import argparse
import io
import json
import re
import tarfile
from datetime import UTC, datetime

from validation.releases import ROOT, digest, exclusive, verify_history


def capture(root, name, note):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("snapshot label must be a plain identifier")
    paths = [
        *sorted((root / "plots").glob("*.py")),
        *sorted((root / "release_tools").glob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "EXPERIMENT_VALIDATION.md",
        root / "tests/test_publication.py",
        root / "experiments/io.py",
    ]
    payload = {str(p.relative_to(root)): p.read_bytes() for p in paths}
    hashes = {name: digest(data) for name, data in payload.items()}
    history = root / "release_history"
    seal_path = history / "report_seals" / f"{name}.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text())
        if hashes != seal["source_sha256"]:
            raise ValueError("review snapshot is frozen; use a new label")
        verify_history(root)
        entries = (history / "history.jsonl").read_text().splitlines()
        if not any(
            digest(line.encode()) == seal["history_entry_sha256"] for line in entries
        ):
            raise ValueError("reporting seal is not anchored in release history")
        return seal
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, data in payload.items():
            member = tarfile.TarInfo(path)
            member.size, member.mode = len(data), 0o444
            archive.addfile(member, io.BytesIO(data))
    archive_hash = digest(buffer.getvalue())
    destination = history / "snapshots" / f"{archive_hash}.tar"
    if not destination.exists():
        exclusive(destination, buffer.getvalue())
    elif digest(destination.read_bytes()) != archive_hash:
        raise ValueError("snapshot archive is corrupt")
    previous = verify_history(root) if (history / "history.jsonl").exists() else ""
    entry = dict(
        release=name,
        kind="reporting",
        archive_sha256=archive_hash,
        captured_at=datetime.now(UTC).isoformat(),
        note=note,
        previous_entry_sha256=previous,
    )
    line = json.dumps(entry, sort_keys=True)
    with (history / "history.jsonl").open("a") as stream:
        stream.write(line + "\n")
    seal = dict(entry, source_sha256=hashes, history_entry_sha256=digest(line.encode()))
    exclusive(seal_path, (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode())
    verify_history(root)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    seal = capture(ROOT, args.name, args.note)
    print(f"{args.name}: {seal['archive_sha256']}")


if __name__ == "__main__":
    main()
