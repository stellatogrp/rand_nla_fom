import pytest

from experiments.batch import task_ids


def test_batches_cover_each_configuration_once():
    groups = [list(task_ids(5697, i, 32)) for i in range(179)]
    assert [i for group in groups for i in group] == list(range(5697))
    assert max(map(len, groups)) == 32
    assert len(groups[-1]) == 1


@pytest.mark.parametrize("args", [(0, 0, 1), (3, -1, 1), (3, 0, 0), (3, 1, 3)])
def test_batch_rejects_invalid_selection(args):
    with pytest.raises(ValueError):
        task_ids(*args)


def test_worker_failure_does_not_skip_later_configurations(tmp_path, monkeypatch):
    import json

    from experiments import batch
    from experiments.io import write_csv

    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, [dict(config=json.dumps({}), run_id=str(i)) for i in range(3)])
    commands = []

    class Child:
        def __init__(self, command):
            commands.append(command)

        def wait(self):
            return 1 if len(commands) == 2 else 0

    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setattr(batch.subprocess, "Popen", Child)
    monkeypatch.setattr(batch.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        batch.sys,
        "argv",
        [
            "batch",
            "--manifest",
            str(manifest),
            "--batch-id",
            "0",
            "--batch-size",
            "3",
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        batch.main()
    assert stopped.value.code == 1
    assert [c[c.index("--task-id") + 1] for c in commands] == ["0", "1", "2"]
