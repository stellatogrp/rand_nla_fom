"""Create content-addressed source archives and refuse changes to sealed releases."""

import argparse
import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources(root, release):
    directory = root / "releases" / release
    if directory.parent != root / "releases" or not directory.is_dir():
        raise ValueError("release must name an existing direct child of releases")
    paths = [
        *directory.glob("experiments/*.py"),
        directory / "README.md",
        directory / "source_manifest.json",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "slurm/common.sh",
        root / "slurm/job_array.slurm",
        root / "slurm/cluster.env",
        root / "slurm/validation_v6.slurm",
        *list((root / "validation").glob("*.py")),
    ]
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(paths)}


def exclusive(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    path.chmod(0o444)


def verify(root, release):
    history = root / "release_history"
    seal = json.loads((history / "seals" / f"{release}.json").read_text())
    archive = history / "snapshots" / f"{seal['archive_sha256']}.tar"
    if digest(archive.read_bytes()) != seal["archive_sha256"]:
        raise ValueError("snapshot archive hash mismatch")
    current = {name: digest(data) for name, data in sources(root, release).items()}
    if current != seal["source_sha256"]:
        raise ValueError("sealed source changed; create a new release instead")
    with tarfile.open(archive) as bundle:
        archived = {
            m.name: digest(bundle.extractfile(m).read()) for m in bundle.getmembers()
        }
    if archived != current:
        raise ValueError("snapshot contents do not match seal")
    verify_history(root)
    entries = (history / "history.jsonl").read_text().splitlines()
    if not any(
        digest(line.encode()) == seal["history_entry_sha256"]
        and json.loads(line)["archive_sha256"] == seal["archive_sha256"]
        and json.loads(line)["release"] == release
        for line in entries
    ):
        raise ValueError("seal is not anchored in release history")
    return seal


def verify_history(root):
    previous = ""
    for line in (root / "release_history/history.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["previous_entry_sha256"] != previous:
            raise ValueError("release history chain is broken")
        archive = root / "release_history/snapshots" / f"{entry['archive_sha256']}.tar"
        if digest(archive.read_bytes()) != entry["archive_sha256"]:
            raise ValueError("historical archive changed")
        previous = digest(line.encode())
    return previous


def freeze(root, release, note):
    history = root / "release_history"
    seal_path = history / "seals" / f"{release}.json"
    if seal_path.exists():
        return verify(root, release)
    payload = sources(root, release)
    # A source_manifest is an additional consistency check, not the historical anchor.
    declared = json.loads(payload[f"releases/{release}/source_manifest.json"])
    expected_names = {
        str(p.relative_to(root / "releases" / release))
        for p in (root / "releases" / release / "experiments").glob("*.py")
    }
    if set(declared["source_sha256"]) != expected_names:
        raise ValueError("source manifest does not cover every Python source")
    for name, expected in declared["source_sha256"].items():
        if digest(payload[f"releases/{release}/{name}"]) != expected:
            raise ValueError(f"source manifest mismatch: {name}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        for name, data in payload.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o444
            bundle.addfile(info, io.BytesIO(data))
    archive_hash = digest(buffer.getvalue())
    archive = history / "snapshots" / f"{archive_hash}.tar"
    if archive.exists():
        if digest(archive.read_bytes()) != archive_hash:
            raise ValueError("existing snapshot is corrupt")
    else:
        exclusive(archive, buffer.getvalue())
    previous = verify_history(root) if (history / "history.jsonl").exists() else ""
    entry = dict(
        release=release,
        archive_sha256=archive_hash,
        captured_at=datetime.now(UTC).isoformat(),
        note=note,
        previous_entry_sha256=previous,
    )
    line = json.dumps(entry, sort_keys=True)
    with (history / "history.jsonl").open("a") as stream:
        stream.write(line + "\n")
    seal = dict(
        entry,
        source_sha256={name: digest(data) for name, data in payload.items()},
        history_entry_sha256=digest(line.encode()),
    )
    exclusive(seal_path, (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode())
    return verify(root, release)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "verify"])
    parser.add_argument("--release", default="accounting_v6")
    parser.add_argument(
        "--note",
        default="Capture current source; earlier review states are not reconstructed.",
    )
    args = parser.parse_args()
    seal = (
        freeze(ROOT, args.release, args.note)
        if args.action == "freeze"
        else verify(ROOT, args.release)
    )
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
