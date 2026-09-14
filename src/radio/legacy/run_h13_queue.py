#!/usr/bin/env python3
"""Finite dependency queue; Linux server only, no scheduling service installed."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def validate_plan(plan):
    jobs = plan["jobs"]
    ids = [j["id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate job ids")
    for job in jobs:
        if not job["id"].replace("-", "").replace("_", "").isalnum():
            raise ValueError("unsafe job id")
        if not isinstance(job["command"], list) or not job["command"]:
            raise ValueError("command must be a nonempty argv array")
        if job.get("kind", "cpu") not in ("cpu", "gpu"):
            raise ValueError("invalid resource kind")
        if not set(job.get("depends", [])).issubset(ids):
            raise ValueError("unknown dependency")
        if not 1 <= job.get("timeout_seconds", 7200) <= 7200:
            raise ValueError("timeout outside protocol cap")
    pending = {j["id"]: set(j.get("depends", [])) for j in jobs}
    while pending:
        ready = {name for name, deps in pending.items() if not deps}
        if not ready:
            raise ValueError("dependency cycle")
        pending = {name: deps - ready for name, deps in pending.items() if name not in ready}


def acquire_gpu(candidates, minimum_mib, lock_root):
    import fcntl
    info = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    free = {int(line.split(",")[0]): int(line.split(",")[1]) for line in info.splitlines()}
    for gpu in sorted(candidates, key=lambda g: -free.get(g, 0)):
        if free.get(gpu, 0) < minimum_mib:
            continue
        lock = open(Path(lock_root) / f"h13-gpu-{gpu}.lock", "a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return gpu, lock
        except BlockingIOError:
            lock.close()
    return None


def run_job(job, root, plan_hash, gpu=None):
    started = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    record = {"id": job["id"], "status": "RUNNING", "plan_sha256": plan_hash,
              "job_sha256": digest(job), "started_unix": started,
              "kind": job.get("kind", "cpu"), "gpu": gpu}
    target = root / "jobs" / (job["id"] + ".json")
    atomic_json(target, record)
    rc = None
    log_path = root / "logs" / (job["id"] + ".log")
    for attempt in range(2):
        record["attempts"] = attempt + 1
        with log_path.open("ab") as log:
            process = subprocess.Popen(job["command"], cwd=job.get("cwd"), env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = process.pid
            atomic_json(target, record)
            try:
                rc = process.wait(timeout=job.get("timeout_seconds", 7200))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                rc = 124
        tail = log_path.read_bytes()[-16000:].decode("utf-8", errors="replace").lower()
        transient = any(x in tail for x in ("cuda out of memory", "resource temporarily unavailable", "device is busy"))
        if rc == 0 or not transient or attempt == 1:
            break
        time.sleep(10)
    record.update(status="COMPLETE" if rc == 0 else "FAILED", exit_code=rc,
                  elapsed_seconds=time.time() - started, finished_unix=time.time())
    atomic_json(target, record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    if a.validate_only:
        print(json.dumps({"status": "PASS", "jobs": len(plan["jobs"])}))
        return
    import fcntl
    root = a.output_dir.resolve()
    for part in ("jobs", "logs", "locks"):
        (root / part).mkdir(parents=True, exist_ok=True)
    gpu_lock_root = Path("/tmp") / f"campuswave-gpu-locks-{os.getuid()}"
    gpu_lock_root.mkdir(mode=0o700, exist_ok=True)
    queue_lock = open(root / "queue.lock", "a+")
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_hash = digest(plan)
    manifest = root / "frozen_plan.json"
    if manifest.exists() and digest(json.loads(manifest.read_text())) != plan_hash:
        raise RuntimeError("changed plan requires a fresh queue output directory")
    atomic_json(manifest, plan)
    jobs = {j["id"]: j for j in plan["jobs"]}
    done, failed = {}, {}
    for name, job in jobs.items():
        path = root / "jobs" / (name + ".json")
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("job_sha256") != digest(job) or old.get("plan_sha256") != plan_hash:
                raise RuntimeError("resume contract mismatch")
            if old["status"] == "COMPLETE":
                done[name] = old
            elif old["status"] in ("FAILED", "BLOCKED"):
                failed[name] = old
            elif old["status"] == "RUNNING" and old.get("pid"):
                try:
                    os.kill(old["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("previous child still alive; refusing duplicate execution")
    pending = {name: job for name, job in jobs.items() if name not in done and name not in failed}
    active = {}
    wait_since = {}
    start = time.time()
    cpu_limit = min(int(plan.get("cpu_workers", 8)), 8)
    gpu_candidates = plan.get("gpu_candidates", [1, 3])
    with ThreadPoolExecutor(max_workers=cpu_limit + len(gpu_candidates)) as pool:
        while pending or active:
            for name, (future, lock) in list(active.items()):
                if future.done():
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"id": name, "status": "FAILED", "reason": str(exc),
                                  "plan_sha256": plan_hash, "job_sha256": digest(jobs[name])}
                        atomic_json(root / "jobs" / (name + ".json"), result)
                    (done if result["status"] == "COMPLETE" else failed)[name] = result
                    if lock:
                        lock.close()
                    del active[name]
            for name, job in list(pending.items()):
                deps = set(job.get("depends", []))
                if deps & set(failed):
                    result = {"id": name, "status": "BLOCKED", "reason": "failed_dependency",
                              "plan_sha256": plan_hash, "job_sha256": digest(job)}
                    failed[name] = result
                    atomic_json(root / "jobs" / (name + ".json"), result)
                    del pending[name]
                    continue
                if not deps.issubset(done):
                    continue
                gpu, lock = None, None
                if job.get("kind", "cpu") == "gpu":
                    wait_since.setdefault(name, time.time())
                    resource = acquire_gpu(gpu_candidates, int(job.get("min_free_mib", 12000)), gpu_lock_root)
                    if resource is None:
                        if time.time() - wait_since[name] > 1800 and not any(jobs[n].get("kind") == "gpu" for n in active):
                            failed[name] = {"id": name, "status": "FAILED", "reason": "resource_wait_timeout",
                                            "plan_sha256": plan_hash, "job_sha256": digest(job)}
                            atomic_json(root / "jobs" / (name + ".json"), failed[name])
                            del pending[name]
                        continue
                    gpu, lock = resource
                elif sum(jobs[n].get("kind", "cpu") == "cpu" for n in active) >= cpu_limit:
                    continue
                active[name] = (pool.submit(run_job, job, root, plan_hash, gpu), lock)
                del pending[name]
            timings = {}
            for name, value in done.items():
                timings.setdefault(jobs[name].get("eta_group", jobs[name].get("kind", "cpu")), []).append(value.get("elapsed_seconds", 0))
            estimates = {group: sum(values) / len(values) for group, values in timings.items() if values}
            remaining_work = {"cpu": 0.0, "gpu": 0.0}
            unknown = []
            for name in list(pending) + list(active):
                job = jobs[name]
                group = job.get("eta_group", job.get("kind", "cpu"))
                if group not in estimates:
                    unknown.append(group)
                    continue
                cost = estimates[group]
                progress_path = root / "jobs" / (name + ".json")
                if name in active and progress_path.exists():
                    progress = json.loads(progress_path.read_text())
                    cost = max(0, cost - (time.time() - progress.get("started_unix", time.time())))
                remaining_work[job.get("kind", "cpu")] += cost
            partial_eta = max(remaining_work["cpu"] / max(cpu_limit, 1), remaining_work["gpu"] / max(len(gpu_candidates), 1))
            status = {"updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.time() - start,
                      "complete": len(done), "failed_or_blocked": len(failed), "total": len(jobs),
                      "active": list(active), "pending": list(pending), "mean_completed_seconds_by_group": estimates,
                      "eta_remaining_seconds_lower_bound": partial_eta,
                      "eta_uncalibrated_groups": sorted(set(unknown)),
                      "eta_note": "throughput estimate; dependencies and shared compute can increase wall time",
                      "status": "RUNNING" if pending or active else ("FAILED" if failed else "COMPLETE")}
            atomic_json(root / "queue.state.json", status)
            if pending or active:
                time.sleep(3)
    print(json.dumps(status))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
