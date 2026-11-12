"""Campaigns of original-format studies on Slurm, one array task per run.

    python -m studies.campaign prepare studies_v1/campaign.json
    python -m studies.campaign task --manifest studies_v1/inputs/qp.run.part000.csv \\
        --task-id 0                                   # what one array task runs
    python -m studies.campaign status studies_v1      # finished runs per study

A campaign specification (JSON) lists studies, each with its study module
(``admm_qp``, ``admm_lasso``, or ``ppm_logistic``), its flags, and its lane.
``prepare`` writes, under ``<campaign>/inputs/``:

* one argument file per study (``<name>.args``), the campaign's common flags
  first, so every stage of a study parses exactly the same settings;
* per lane, a calibrate manifest (one task per study, which also sets up
  every instance), run manifests (one task per study and configuration, which
  runs it for every instance), and a collect manifest (one task per study);
* ``seal.json``: the SHA256 of the specification, every input, the
  ``studies`` sources, and the Slurm script, and the sealed engine release.
  Every task verifies it first, so code or inputs changed after a launch
  cannot mix into a running campaign; a changed campaign needs a new name.

Lanes keep the scheduling caps of ``AGENTS.md`` apart (eight QP tasks, one
Kaczmarz task, two regression tasks): each lane has its own calibrate array,
run array, and collect job, so a slow lane delays no other.  Within a lane every
configuration of every study is its own task, so a configuration that stalls
holds one slot until its time limit and delays no other result; its instances
run in turn, each stored as it finishes.  (One task per run would exceed
Della's per-user submission limit of 1,000 jobs, ``SUBMIT_CAP``, many times
over.)  The run manifests are configuration-major, so the baselines of every
study come first.  A run manifest longer than Della's array limit is split into
parts that run side by side, the lane's cap divided among them.  Each study
reserves room for its recommended runs (``sigma*`` and two ``s*``), which are
known only after calibration; a reserved index beyond a study's list is a
no-op task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from . import ENGINE
from .study import (
    CONFIGURATIONS,
    PARTS,
    ROOT,
    _complete,
    _read_configurations,
    _run_files,
    _verify_engine,
    _write_text_rows,
    build_parser,
    run_calibration,
    run_collect,
    run_run,
    settings_from,
    study_directory,
)

MODULES = ("admm_qp", "admm_lasso", "ppm_logistic")
SCRIPT = "slurm/studies_campaign.slurm"
# Della's MaxArraySize is 2501 (``slurm_agent.py recommend``): indices 0-2500.
MAX_ARRAY = 2500
# Della's per-user submission limit of its CPU QoS (``slurm_agent.py discover``):
# every array task counts, pending or running, so a campaign stays below it.
SUBMIT_CAP = 1000
# Inner solvers without verified constants have no sigma* and s* runs.
UNCONSTRAINED = ("agd", "nystrom_pcg", "cyclic_kaczmarz")
MANIFEST_FIELDS = (
    "task",
    "stage",
    "lane",
    "study",
    "module",
    "args",
    "instance",
    "configuration",
)


def _relative(path: Path) -> str:
    """A path as recorded in the inputs: relative to the project, as on Della.

    A campaign outside the project (a test's) records absolute paths.
    """
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_rows(path, rows)


def _read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def reserved_configurations(settings: dict) -> int:
    """Runs per instance, counting the recommended ones that may be added.

    The two baselines, the sigma sweep, and the stride sweep, plus ``sigma*`` and
    up to two ``s*`` strides off the grid when the inner solver has constants.
    """
    count = 2 + len(settings["sigmas"]) + len(settings["bs"])
    unconstrained = (
        settings["algorithm"] == "admm" and settings["inner"] in UNCONSTRAINED
    )
    return count if unconstrained else count + 3


def _parse(module: str, argv: list[str]) -> argparse.Namespace:
    if module not in MODULES:
        raise SystemExit(f"unknown study module {module!r}")
    return build_parser(module).parse_args(argv)


def prepare(spec_path: Path, *, force: bool = False) -> Path:
    """Write the inputs and the seal of the campaign in ``spec_path``."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    directory = spec_path.parent
    inputs = directory / "inputs"
    seal_path = directory / "seal.json"
    if seal_path.exists() and not force:
        raise SystemExit(
            f"{seal_path} exists: a sealed campaign is frozen once launched; "
            "use a new campaign name, or --force before any launch"
        )
    for stale in inputs.glob("*") if inputs.exists() else []:
        stale.unlink()
    inputs.mkdir(parents=True, exist_ok=True)
    common = list(spec.get("common", []))
    results_dir = spec["results_dir"]
    lanes = spec["lanes"]
    studies, names = [], set()
    for entry in spec["studies"]:
        module, lane = entry["module"], entry["lane"]
        if lane not in lanes:
            raise SystemExit(f"study lane {lane!r} is not among {sorted(lanes)}")
        flags = [
            *common,
            *lanes[lane].get("flags", []),
            *entry.get("flags", []),
            "--results-dir",
            results_dir,
        ]
        settings = settings_from(_parse(module, flags), module)
        name = settings["name"]
        if name in names:
            raise SystemExit(f"two studies share the name {name!r}; give --name")
        names.add(name)
        args_path = inputs / f"{name}.args"
        lines = [f"# {name}: {module}, lane {lane}", *_lines(flags, name)]
        args_path.write_text("\n".join(lines) + "\n")
        studies.append(
            dict(
                name=name,
                module=module,
                lane=lane,
                args=_relative(args_path),
                instances=int(settings["instances"]),
                reserved=reserved_configurations(settings),
            )
        )
    plan = {}
    for lane, options in lanes.items():
        members = [s for s in studies if s["lane"] == lane]
        if not members:
            continue
        plan[lane] = _lane_manifests(inputs, lane, members, int(options["throttle"]))
    queued = sum(
        info["tasks"] + 2 * len([s for s in studies if s["lane"] == lane])
        for lane, info in plan.items()
    )
    if queued > SUBMIT_CAP:
        print(
            f"warning: {queued} tasks exceed Della's per-user submission limit of "
            f"{SUBMIT_CAP}; submit the lanes one after another",
            file=sys.stderr,
        )
    sources = {
        _relative(path): _sha256(path)
        for path in sorted(
            [
                *(
                    p.resolve()
                    for p in directory.rglob("*")
                    # Notes (README, launch status) stay editable.
                    if p.is_file() and p.name != "seal.json" and p.suffix != ".md"
                ),
                *(ROOT / "studies").glob("*.py"),
                ROOT / SCRIPT,
            ]
        )
    }
    seal = dict(
        campaign=spec["campaign"],
        engine=ENGINE,
        engine_archive_sha256=_verify_engine(),
        results_dir=results_dir,
        lanes={lane: dict(lanes[lane], **plan[lane]) for lane in plan},
        studies=studies,
        sources=sources,
    )
    seal_path.write_text(json.dumps(seal, indent=1, sort_keys=True) + "\n")
    return seal_path


def _lines(flags: list[str], name: str) -> list[str]:
    """The flags one option per line, with the study's name pinned."""
    lines, current = [], []
    for flag in [*flags, "--name", name]:
        if flag.startswith("--") and current:
            lines.append(" ".join(current))
            current = []
        current.append(flag)
    if current:
        lines.append(" ".join(current))
    return lines


def _lane_manifests(
    inputs: Path, lane: str, studies: list[dict], throttle: int
) -> dict:
    """The calibrate, run, and collect manifests of one lane."""

    def row(stage, study, instance="", configuration=""):
        return dict(
            stage=stage,
            lane=lane,
            study=study["name"],
            module=study["module"],
            args=study["args"],
            instance=instance,
            configuration=configuration,
        )

    calibrate = [row("calibrate", s) for s in studies]
    collect = [row("collect", s) for s in studies]
    runs = [
        row("run", s, "", configuration)
        for configuration in range(max(s["reserved"] for s in studies))
        for s in studies
        if configuration < s["reserved"]
    ]
    names = {}
    for stage, rows in (("calibrate", calibrate), ("collect", collect)):
        path = inputs / f"{lane}.{stage}.csv"
        _write_rows(path, [dict(task=i, **r) for i, r in enumerate(rows)])
        names[stage] = _relative(path)
    parts = max(1, math.ceil(len(runs) / (MAX_ARRAY + 1)))
    size = math.ceil(len(runs) / parts)
    shards = []
    for part in range(parts):
        chunk = runs[part * size : (part + 1) * size]
        path = inputs / f"{lane}.run.part{part:03d}.csv"
        _write_rows(path, [dict(task=i, **r) for i, r in enumerate(chunk)])
        shards.append(dict(manifest=_relative(path), tasks=len(chunk)))
    # Parts run side by side with the lane's cap divided among them, rather
    # than one after another, where a slow task would delay the next part.
    per_part = max(1, throttle // parts)
    return dict(
        calibrate=names["calibrate"],
        collect=names["collect"],
        runs=shards,
        run_throttle=per_part,
        run_parts_chained=throttle < parts,
        tasks=len(runs),
    )


def verify(manifest: Path) -> dict:
    """Check the campaign seal of ``manifest``; return the seal."""
    directory = Path(manifest).resolve().parent.parent
    seal_path = directory / "seal.json"
    if not seal_path.exists():
        raise SystemExit(f"{seal_path} is missing: prepare the campaign first")
    seal = json.loads(seal_path.read_text())
    changed = [
        name
        for name, digest in seal["sources"].items()
        if not (ROOT / name).exists() or _sha256(ROOT / name) != digest
    ]
    if changed:
        raise SystemExit(
            f"campaign {seal['campaign']} is sealed and these changed since: "
            f"{', '.join(changed)}; prepare a new campaign instead"
        )
    archive = _verify_engine()
    if archive != seal["engine_archive_sha256"]:
        raise SystemExit(f"engine {ENGINE} is not the sealed archive")
    return seal


def task(manifest: Path, task_id: int) -> Path:
    """Run row ``task_id`` of a campaign manifest: one array task."""
    seal = verify(manifest)
    rows = _read_rows(manifest)
    if not 0 <= task_id < len(rows) or int(rows[task_id]["task"]) != task_id:
        raise SystemExit(f"task {task_id} is not in {manifest}")
    row = rows[task_id]
    argv = [f"@{ROOT / row['args']}", "--stage", row["stage"]]
    if row["stage"] == "run":
        argv += ["--configuration", row["configuration"]]
        if row["instance"] != "":
            argv += ["--task-index", row["instance"]]
    if row["stage"] == "collect":
        # The lane's runs are over or out of time: collect what finished and
        # list the rest in missing.csv rather than wait for it.
        argv.append("--allow-partial")
    args = _parse(row["module"], argv)
    stage = dict(calibrate=run_calibration, run=run_run, collect=run_collect)
    print(
        f"[{seal['campaign']}] {row['lane']} {row['stage']} {row['study']} "
        f"instance {row['instance'] or '-'} configuration "
        f"{row['configuration'] or '-'}",
        flush=True,
    )
    return stage[row["stage"]](row["module"], args)


HELPER = "python3 ~/.claude/skills/slurm/scripts/slurm_agent.py"


def pilot_task(seal: dict, lane: str, study: str) -> tuple[str, int]:
    """The manifest and index of ``study``'s near-exact configuration.

    The near-exact baseline is the longest run of an instance, so the task that
    runs it for every instance bounds the time and memory of the lane's other
    tasks.
    """
    for part in seal["lanes"][lane]["runs"]:
        for row in _read_rows(ROOT / part["manifest"]):
            if (row["study"], row["configuration"]) == (study, "1"):
                return part["manifest"], int(row["task"])
    raise SystemExit(f"{study} has no near-exact run in lane {lane}")


def commands(campaign: Path, pilots: dict[str, str] | None = None) -> list[str]:
    """The submissions of a campaign, lane by lane, for the slurm skill's helper.

    Each lane: its calibrate array; a pilot run (the near-exact run of the first
    instance of the lane's ``pilots`` study, by default its last study) to size
    the run arrays; the run arrays; and the collect array.  ``CAL_<lane>``,
    ``PILOT_<lane>``, and ``RUN_<lane>_<part>`` stand for the job ids that
    ``submit --yes`` prints.  Every ``submit`` validates with ``--test-only``
    first and submits only with ``--yes``.
    """
    campaign = Path(campaign)
    seal = json.loads((campaign / "seal.json").read_text())
    config = f"slurm/{seal['campaign']}.env"
    base = f"{HELPER} submit {SCRIPT} --config {config} --cpus 1 --constraint icelake"
    lines = []
    for lane, info in seal["lanes"].items():
        members = [s["name"] for s in seal["studies"] if s["lane"] == lane]
        name = f"rnla-{seal['campaign']}-{lane}"
        throttle = int(info["throttle"])
        study = (pilots or {}).get(lane, members[-1])
        manifest, index = pilot_task(seal, lane, study)
        lines += [
            f"# lane {lane}: {len(members)} studies, {info['tasks']} run tasks in "
            f"{len(info['runs'])} part(s), at most {throttle} at a time",
            # Calibrating also sets up every instance, ground truths included.
            f"{base} --mem {info['mem']} "
            f"--time {info.get('calibrate_time', info['time'])} "
            f"--array 0-{len(members) - 1}%{throttle} --job-name {name}-calibrate "
            f"--export MANIFEST={info['calibrate']}",
            f"# pilot: {study}, near-exact configuration, every instance",
            f"{base} --mem {info['mem']} --time {info['time']} --array {index} "
            f"--job-name {name}-pilot --export MANIFEST={manifest} "
            f"--dependency afterok:CAL_{lane}",
        ]
        runs = []
        for number, part in enumerate(info["runs"]):
            runs.append(f"RUN_{lane}_{number}")
            # Chained parts (a cap smaller than the number of parts) wait for
            # the previous part; otherwise the parts share the cap side by side.
            after = (
                f"afterany:RUN_{lane}_{number - 1}"
                if info["run_parts_chained"] and number
                else f"afterany:CAL_{lane}"
            )
            lines.append(
                f"{base} --mem {info['mem']} --time {info['time']} "
                f"--array 0-{part['tasks'] - 1}%{info['run_throttle']} "
                f"--job-name {name}-run{number} --export MANIFEST={part['manifest']} "
                f"--dependency {after}"
            )
        lines.append(
            f"{base} --mem {info['mem']} --time 00:30:00 "
            f"--array 0-{len(members) - 1}%{throttle} --job-name {name}-collect "
            f"--export MANIFEST={info['collect']} "
            f"--dependency afterany:{':'.join(runs)}"
        )
    return lines


def status(campaign: Path) -> list[dict]:
    """Finished and planned runs of every study, from the result files."""
    seal = json.loads((Path(campaign) / "seal.json").read_text())
    rows = []
    for study in seal["studies"]:
        args = _parse(
            study["module"], [f"@{ROOT / study['args']}", "--stage", "collect"]
        )
        settings = settings_from(args, study["module"], pin=False)
        directory = study_directory(settings)
        planned = _read_configurations(directory / CONFIGURATIONS)
        seeds = [int(settings["seed"]) + i for i in range(int(settings["instances"]))]
        finished = sum(
            _run_files(directory / PARTS / str(seed), index)["results"].exists()
            for seed in seeds
            for index in range(len(planned or []))
        )
        rows.append(
            dict(
                lane=study["lane"],
                study=study["name"],
                calibrated=planned is not None,
                planned=len(planned or []) * len(seeds),
                finished=finished,
                instances_complete=sum(_complete(directory, seed) for seed in seeds),
                instances=len(seeds),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subcommands = parser.add_subparsers(dest="command", required=True)
    making = subcommands.add_parser("prepare", help="write the inputs and the seal")
    making.add_argument("spec", type=Path)
    making.add_argument(
        "--force", action="store_true", help="replace a campaign not yet launched"
    )
    running = subcommands.add_parser("task", help="run one row of a manifest")
    running.add_argument("--manifest", type=Path, required=True)
    running.add_argument("--task-id", type=int, required=True)
    running.add_argument(
        "--allow-local",
        action="store_true",
        help="compute outside a Slurm allocation (workstations, never login nodes)",
    )
    showing = subcommands.add_parser("status", help="finished runs per study")
    showing.add_argument("campaign", type=Path)
    listing = subcommands.add_parser("commands", help="print the submissions")
    listing.add_argument("campaign", type=Path)
    listing.add_argument(
        "--pilot",
        nargs=2,
        action="append",
        metavar=("LANE", "STUDY"),
        help="the study whose near-exact run pilots a lane",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.spec, force=args.force))
    elif args.command == "task":
        if not (os.environ.get("SLURM_JOB_ID") or args.allow_local):
            parser.error("campaign tasks run inside a Slurm allocation")
        task(args.manifest, args.task_id)
    elif args.command == "commands":
        print("\n".join(commands(args.campaign, dict(args.pilot or []))))
    else:
        rows = status(args.campaign)
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
