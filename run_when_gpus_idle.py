#!/usr/bin/env python3
"""Wait for idle GPUs, then run one argv command without changing its settings.

Example: python run_when_gpus_idle.py --gpus 4,5 -- ./some_experiment.sh
Use --wait-only to observe availability without launching a command.
Locks coordinate this launcher only; they cannot reserve GPUs against other tools.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
HOST = socket.gethostname().split(".")[0]
LOCK_ROOT = ROOT / "logs" / "gpu_queue" / HOST / "locks"
ROUTING_FILE = ROOT / "logs" / "gpu_queue" / "assignments.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def gpu_indices(value):
    try:
        indices = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use physical GPU indices, e.g. 4,5") from error
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("GPU indices must be distinct and nonnegative")
    return indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=gpu_indices, default=gpu_indices("4,5"))
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--idle-seconds", type=float, default=30)
    parser.add_argument("--log-dir", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--routing-file", type=Path, default=ROUTING_FILE,
                        help="Optional JSON GPU assignments, matched against the complete command argv")
    parser.add_argument("--wait-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if bool(args.command) == args.wait_only:
        parser.error("Provide a command after --, or explicitly select --wait-only")
    if args.poll_seconds <= 0 or args.idle_seconds < 0:
        parser.error("poll-seconds must be positive; idle-seconds must be nonnegative")
    args.cwd = args.cwd.resolve()
    if not args.cwd.is_dir():
        parser.error("cwd must be an existing directory")
    return args


def assigned_route(path, command):
    """Read the whole assignment file before deciding whether this command moved.

    A missing file means no assignments. Malformed or ambiguous assignments are
    errors, including malformed entries unrelated to this command. No command is
    executed while its assignment is unknown.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or not isinstance(data.get("jobs"), list)):
        raise ValueError("Malformed GPU assignment file")
    routes = {}
    for job in data["jobs"]:
        if not isinstance(job, dict):
            raise ValueError("Malformed GPU job assignment")
        argv, host, gpus = job.get("command"), job.get("host"), job.get("gpus")
        if (not isinstance(argv, list) or not argv
                or not all(isinstance(arg, str) and "\0" not in arg for arg in argv)
                or not argv[0] or not isinstance(host, str) or not host.strip()
                or not isinstance(gpus, list) or not gpus
                or not all(type(index) is int and index >= 0 for index in gpus)
                or len(set(gpus)) != len(gpus)):
            raise ValueError("Malformed GPU job command, host or indices")
        key = tuple(argv)
        if key in routes:
            raise ValueError("Duplicate command in GPU assignment file")
        routes[key] = job
    return routes.get(tuple(command))


def nvidia_query(fields, kind="gpu"):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return [[field.strip() for field in row] for row in csv.reader(result.stdout.split("\n")) if row]


def gpu_snapshot(indices):
    selected = {}
    for row in nvidia_query("index,uuid,memory.used,utilization.gpu"):
        if len(row) != 4:
            raise RuntimeError("Malformed NVIDIA GPU query; availability is unknown")
        index = int(row[0])
        if index in indices:
            if index in selected:
                raise RuntimeError("NVIDIA query returned a duplicate GPU")
            selected[index] = {"index": index, "uuid": row[1],
                               "memory_used_mib": int(row[2]), "utilization_percent": int(row[3])}
    if set(selected) != set(indices):
        raise RuntimeError("Not all requested GPUs are present")
    uuids = {row["uuid"] for row in selected.values()}
    processes = []
    for row in nvidia_query("gpu_uuid,pid", "compute-apps"):
        if len(row) != 2:
            raise RuntimeError("Malformed NVIDIA process query; availability is unknown")
        if row[0] in uuids:
            processes.append({"gpu_uuid": row[0], "pid": int(row[1])})
    idle = not processes and all(
        row["memory_used_mib"] < 1024 and row["utilization_percent"] <= 10
        for row in selected.values()
    )
    return {"idle": idle, "gpus": [selected[i] for i in indices], "compute_processes": processes}


@contextmanager
def gpu_locks(indices):
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for index in sorted(indices):
            handle = stack.enter_context((LOCK_ROOT / f"gpu_{index}.lock").open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another idle-GPU launcher already owns GPU {index} on {HOST}") from error
        yield


class Reporter:
    def __init__(self, directory, manifest):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.log = (directory / "run.log").open("a", buffering=1)
        self.wait_only = manifest["wait_only"]

    def update(self, status, **details):
        record = {"updated_utc": utc_now(), "status": status, "wait_only": self.wait_only, **details}
        temporary = self.directory / "status.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(self.directory / "status.json")
        message = json.dumps(record)
        print(message, flush=True)
        self.log.write(message + "\n")


def wait_for_idle(args, reporter):
    idle_since, observations = None, 0
    while True:
        snapshot = gpu_snapshot(args.gpus)
        now = time.monotonic()
        if snapshot["idle"]:
            idle_since = now if idle_since is None else idle_since
            observations += 1
        else:
            idle_since, observations = None, 0
        elapsed = 0 if idle_since is None else now - idle_since
        ready = snapshot["idle"] and observations >= 2 and elapsed >= args.idle_seconds
        reporter.update("ready" if ready else "waiting_for_gpus", idle_for_seconds=elapsed, **snapshot)
        if ready:
            return
        interval = args.poll_seconds
        if snapshot["idle"]:
            interval = min(interval, max(0.01, args.idle_seconds - elapsed))
        time.sleep(interval)


def run_child(args, reporter):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, args.gpus)))
    # No shell: characters such as $, backticks and spaces remain literal argv.
    child = subprocess.Popen(args.command, cwd=args.cwd, env=env,
                             stdout=reporter.log, stderr=subprocess.STDOUT, start_new_session=True)
    reporter.update("running", child_pid=child.pid, command=args.command)
    old_handlers = {}

    def forward(signum, frame):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, forward)
        code = child.wait()
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    code = code if code >= 0 else 128 - code
    reporter.update("completed" if code == 0 else "failed", exit_code=code)
    return code


def main(argv=None):
    args = parse_args(argv)
    directory = args.log_dir or ROOT / "logs" / "gpu_queue" / HOST / (
        "gpu_" + "_".join(map(str, args.gpus)) + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{os.getpid()}"
    )
    reporter = Reporter(directory.resolve(), {"created_utc": utc_now(), "host": HOST,
        "gpus": args.gpus, "cwd": str(args.cwd), "command": args.command, "wait_only": args.wait_only,
        "poll_seconds": args.poll_seconds, "idle_seconds": args.idle_seconds,
        "lock_root": str(LOCK_ROOT), "routing_file": str(args.routing_file.resolve()), "pid": os.getpid()})
    print(f"Queue output: {reporter.directory}", flush=True)
    try:
        route = assigned_route(args.routing_file, args.command)
        if route is not None and (route["host"] != HOST or route["gpus"] != args.gpus):
            reporter.update("reassigned", assigned_host=route["host"], assigned_gpus=route["gpus"],
                            command=args.command, exit_code=0)
            return 0
        with gpu_locks(args.gpus):
            wait_for_idle(args, reporter)
            if args.wait_only:
                reporter.update("completed", wait_only=True, exit_code=0)
                return 0
            return run_child(args, reporter)
    except KeyboardInterrupt:
        reporter.update("cancelled", exit_code=130)
        return 130
    except Exception as error:
        reporter.update("failed", error=f"{type(error).__name__}: {error}", exit_code=1)
        return 1
    finally:
        reporter.log.close()


if __name__ == "__main__":
    sys.exit(main())
