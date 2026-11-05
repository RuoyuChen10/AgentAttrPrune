"""CPU checks for GPU availability gating and safe command dispatch."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run_when_gpus_idle.py"
spec = importlib.util.spec_from_file_location("idle_queue", SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class Clock:
    def __init__(self):
        self.now = 0.0

    def sleep(self, seconds):
        self.now += seconds

    def monotonic(self):
        return self.now


class Reporter:
    def __init__(self):
        self.events = []

    def update(self, status, **details):
        self.events.append({"status": status, **details})


@pytest.fixture(autouse=True)
def isolated_routing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROUTING_FILE", tmp_path / "assignments.json")


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(queue.time, "monotonic", value.monotonic)
    monkeypatch.setattr(queue.time, "sleep", value.sleep)
    return value


def query_fixture(monkeypatch, gpu_rows, apps):
    def query(fields, kind="gpu"):
        return gpu_rows if kind == "gpu" else apps
    monkeypatch.setattr(queue, "nvidia_query", query)


def test_compute_process_blocks_even_when_utilization_and_memory_low(monkeypatch):
    query_fixture(monkeypatch, [["4", "gpu4", "7", "0"], ["5", "gpu5", "7", "0"]],
                  [["gpu4", "123"]])
    state = queue.gpu_snapshot([4, 5])
    assert state["idle"] is False
    assert state["compute_processes"] == [{"gpu_uuid": "gpu4", "pid": 123}]


def test_busy_observation_restarts_continuous_idle_window(monkeypatch, clock):
    states = iter([True, False, True, True])
    monkeypatch.setattr(queue, "gpu_snapshot", lambda _: {"idle": next(states)})
    reporter = Reporter()
    queue.wait_for_idle(SimpleNamespace(gpus=[4, 5], poll_seconds=30, idle_seconds=30), reporter)
    assert clock.now == 90
    assert [event["status"] for event in reporter.events] == [
        "waiting_for_gpus", "waiting_for_gpus", "waiting_for_gpus", "ready"]
    assert reporter.events[-1]["idle_for_seconds"] == 30


@pytest.mark.parametrize("problem", ["missing", "unknown_memory"])
def test_unknown_gpu_state_never_counts_as_idle(monkeypatch, problem):
    rows = [["4", "gpu4", "7", "0"]]
    if problem == "unknown_memory":
        rows += [["5", "gpu5", "N/A", "0"]]
    query_fixture(monkeypatch, rows, [])
    with pytest.raises((ValueError, RuntimeError)):
        queue.gpu_snapshot([4, 5])


def test_query_failure_records_failure_without_starting_command(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "LOCK_ROOT", tmp_path / "locks")
    def broken_query(_):
        raise subprocess.CalledProcessError(1, ["nvidia-smi"])
    monkeypatch.setattr(queue, "gpu_snapshot", broken_query)
    def forbidden(*args):
        pytest.fail("A command must not launch when GPU status is unknown")
    monkeypatch.setattr(queue, "run_child", forbidden)
    out = tmp_path / "queue"
    assert queue.main(["--log-dir", str(out), "--", "unreachable-command"]) == 1
    assert json.loads((out / "status.json").read_text())["status"] == "failed"


def test_overlapping_gpu_queues_conflict_and_release_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "LOCK_ROOT", tmp_path / "locks")
    with queue.gpu_locks([4, 5]):
        with pytest.raises(RuntimeError, match="GPU 5"):
            with queue.gpu_locks([5, 6]):
                pytest.fail("Overlapping GPU set acquired")
        with queue.gpu_locks([6, 7]):
            pass
    with queue.gpu_locks([5, 6]):
        pass


def test_command_argv_env_cwd_exit_and_lock_held_through_child(tmp_path, monkeypatch, clock):
    lock_root = tmp_path / "shared_locks"
    monkeypatch.setattr(queue, "LOCK_ROOT", lock_root)
    monkeypatch.setattr(queue, "gpu_snapshot", lambda _: {"idle": True})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    out, received = tmp_path / "custom_logs", tmp_path / "received.json"
    literals = ["$HOME; echo x", "`touch should_not_exist`", "space and\nnewline"]
    program = (
        "import fcntl,json,os,sys; from pathlib import Path\n"
        "with open(sys.argv[2], 'r+') as h:\n"
        " try:\n  fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB); held=False\n"
        " except BlockingIOError:\n  held=True\n"
        "Path(sys.argv[1]).write_text(json.dumps({'args':sys.argv[3:],'gpus':os.environ['CUDA_VISIBLE_DEVICES'],"
        "'cwd':os.getcwd(),'held':held}))\n"
        "print('child output retained',flush=True)\n"
        "sys.exit(9)\n"
    )
    command = [sys.executable, "-c", program, str(received), str(lock_root / "gpu_4.lock"), *literals]
    code = queue.main(["--gpus", "4,5", "--cwd", str(tmp_path), "--idle-seconds", "0",
                       "--log-dir", str(out), "--", *command])
    assert code == 9
    assert json.loads(received.read_text()) == {
        "args": literals, "gpus": "4,5", "cwd": str(tmp_path), "held": True}
    assert json.loads((out / "manifest.json").read_text())["command"] == command
    assert json.loads((out / "status.json").read_text())["exit_code"] == 9
    assert "child output retained" in (out / "run.log").read_text()
    assert not (tmp_path / "should_not_exist").exists()


def test_wait_only_never_launches_and_command_is_required(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(queue, "LOCK_ROOT", tmp_path / "locks")
    monkeypatch.setattr(queue, "gpu_snapshot", lambda _: {"idle": True})
    monkeypatch.setattr(queue, "run_child", lambda *args: pytest.fail("wait-only launched a command"))
    out = tmp_path / "wait"
    assert queue.main(["--wait-only", "--idle-seconds", "0", "--log-dir", str(out)]) == 0
    assert json.loads((out / "status.json").read_text())["wait_only"] is True
    with pytest.raises(SystemExit):
        queue.parse_args([])
    with pytest.raises(SystemExit):
        queue.parse_args(["--wait-only", "--", "echo", "no"])


@pytest.mark.parametrize("host,gpus", [(queue.HOST, [6, 7]), ("another-host", [4, 5])])
def test_reassigned_command_skips_before_gpu_query_or_lock(tmp_path, monkeypatch, host, gpus):
    command = ["/some/launcher.sh", "all", "--resume"]
    route = tmp_path / "routes.json"
    route.write_text(json.dumps({"schema_version": 1, "jobs": [
        {"command": command, "host": host, "gpus": gpus}]}))
    def forbidden(*args):
        pytest.fail("Transferred jobs must not acquire GPU locks, query GPUs or execute")
    for function in ("gpu_locks", "gpu_snapshot", "run_child"):
        monkeypatch.setattr(queue, function, forbidden)
    out = tmp_path / "queue"
    assert queue.main(["--routing-file", str(route), "--log-dir", str(out), "--", *command]) == 0
    state = json.loads((out / "status.json").read_text())
    assert state["status"] == "reassigned"
    assert state["assigned_host"] == host and state["assigned_gpus"] == gpus


@pytest.mark.parametrize("argv_suffix", [[], ["--different-run"]])
def test_route_match_or_different_full_argv_preserves_execution(tmp_path, monkeypatch, clock, argv_suffix):
    monkeypatch.setattr(queue, "LOCK_ROOT", tmp_path / "locks")
    monkeypatch.setattr(queue, "gpu_snapshot", lambda _: {"idle": True})
    command = ["/some/launcher.sh", "all", "--resume"]
    route = tmp_path / "routes.json"
    route.write_text(json.dumps({"schema_version": 1, "jobs": [
        {"command": command, "host": queue.HOST, "gpus": [6, 7] if argv_suffix else [4, 5]}]}))
    received = []
    monkeypatch.setattr(queue, "run_child", lambda args, reporter: received.append(args.command) or 7)
    assert queue.main(["--routing-file", str(route), "--log-dir", str(tmp_path / "queue"),
                       "--idle-seconds", "0", "--", *command, *argv_suffix]) == 7
    assert received == [command + argv_suffix]


@pytest.mark.parametrize("problem", ["invalid_json", "duplicate", "bad_gpus", "bad_schema"])
def test_malformed_routes_fail_closed(tmp_path, monkeypatch, problem):
    job = {"command": ["/some/launcher.sh"], "host": queue.HOST, "gpus": [6, 7]}
    payload = {"schema_version": 1, "jobs": [job]}
    if problem == "duplicate":
        payload["jobs"].append(dict(job))
    elif problem == "bad_gpus":
        job["gpus"] = [6, 6]
    elif problem == "bad_schema":
        payload["schema_version"] = True
    route = tmp_path / "routes.json"
    route.write_text("{" if problem == "invalid_json" else json.dumps(payload))
    def forbidden(*args):
        pytest.fail("Malformed routes must not access GPUs or run commands")
    for function in ("gpu_locks", "gpu_snapshot", "run_child"):
        monkeypatch.setattr(queue, function, forbidden)
    out = tmp_path / "queue"
    assert queue.main(["--routing-file", str(route), "--log-dir", str(out), "--", "/some/launcher.sh"]) == 1
    assert json.loads((out / "status.json").read_text())["status"] == "failed"
