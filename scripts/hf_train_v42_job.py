# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.2", "bitsandbytes>=0.45", "datasets>=3.2",
#   "huggingface-hub>=1.0", "jinja2>=3.1", "peft>=0.14",
#   "safetensors>=0.4", "torch>=2.6", "transformers>=4.55",
# ]
# ///
from __future__ import annotations

import hashlib, json, os, subprocess, sys, tarfile, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from huggingface_hub import HfApi

INPUT, WORK = Path("/input"), Path("/workspace")
SOURCE, DATA = WORK / "scrabble_ai", WORK / "scrabble_ai/data/general_v42"

def now(): return datetime.now(timezone.utc).isoformat()
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
def run(command, env):
    print("[run] " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=SOURCE, env=env, check=True)
def upload_json(api, repo, payload, repo_path, message):
    path = WORK / Path(repo_path).name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    api.upload_file(path_or_fileobj=str(path), path_in_repo=repo_path, repo_id=repo, commit_message=message)
def evaluate(adapter, dataset, output, env):
    run([sys.executable, str(SOURCE/"scripts/evaluate_v42.py"), "--adapter", str(adapter),
         "--dataset", str(DATA/dataset), "--output", str(output), "--batch-size", "4", "--seed", "7421"], env)
    return json.loads(output.read_text())["summary"]
def train(stage, init, output, env):
    run([sys.executable, str(SOURCE/"scripts/train_qlora.py"), "--model", "Qwen/Qwen3-4B-Instruct-2507",
         "--init-adapter", str(init), "--train-file", str(DATA/stage["file"]),
         "--validation-file", str(DATA/"validation.jsonl"), "--output-dir", str(output),
         "--max-length", "1216", "--max-steps", str(stage["steps"]), "--learning-rate", str(stage["lr"]),
         "--batch-size", "1", "--gradient-accumulation", "8", "--checkpoint-every", "100",
         "--logging-steps", "10", "--save-total-limit", "2", "--skip-final-eval",
         "--no-gradient-checkpointing", "--seed", "7421"], env)
    return json.loads((output/"training_metrics.json").read_text())

def main():
    started = time.time(); manifest = json.loads((INPUT/"job_manifest.json").read_text())
    repo = manifest["output_repo"]; api = HfApi(token=os.environ.get("HF_TOKEN")); api.create_repo(repo, private=True, exist_ok=True)
    status: dict[str, Any] = {"state":"starting", "started_at_utc":now(), "manifest":manifest, "stages":{}}
    env = os.environ.copy()
    try:
        for name, expected in manifest["sha256"].items():
            if sha256(INPUT/name) != expected: raise RuntimeError(f"Input hash mismatch: {name}")
        SOURCE.mkdir(parents=True, exist_ok=True)
        for name in ("source.tar.gz", "data.tar.gz"):
            with tarfile.open(INPUT/name, "r:gz") as archive: archive.extractall(SOURCE, filter="data")
        env["PYTHONPATH"] = str(SOURCE)
        current = INPUT/"init_adapter"
        if not (current/"adapter_model.safetensors").exists(): raise FileNotFoundError(current)
        evaluations = WORK/"evaluations"; evaluations.mkdir()
        status["state"] = "baseline_evaluation"; upload_json(api, repo, status, "status.json", "Start V4.2")
        baseline_path = evaluations/"baseline.json"; status["baseline"] = evaluate(current, "baseline_gate.json", baseline_path, env)
        api.upload_file(path_or_fileobj=str(baseline_path), path_in_repo="evaluations/baseline.json", repo_id=repo, commit_message="V4.2 baseline")
        stopped = None
        for stage in manifest["stages"]:
            name = stage["name"]; status.update(state="training", current_stage=name); upload_json(api, repo, status, "status.json", f"Start {name}")
            output = WORK/name; metrics = train(stage, current, output, env)
            api.upload_folder(folder_path=str(output), path_in_repo=f"{name}/adapter", repo_id=repo,
                allow_patterns=["adapter_config.json","adapter_model.safetensors","tokenizer*","trainer_state.json","training_metrics*.json","segment_metrics.json"],
                commit_message=f"Upload {name} adapter")
            gate_path = evaluations/f"{name}.json"; summary = evaluate(output, stage["gate_file"], gate_path, env)
            api.upload_file(path_or_fileobj=str(gate_path), path_in_repo=f"evaluations/{name}.json", repo_id=repo, commit_message=f"Upload {name} gate")
            passed = all(float(summary["by_task"][task][metric]) >= threshold for task, metric, threshold in stage["gate"])
            status["stages"][name] = {"training":metrics,"evaluation":summary,"gate_passed":passed}; current = output
            upload_json(api, repo, status, "status.json", f"Complete {name}")
            if not passed: stopped = name; break
        status.update(state="complete", completed_at_utc=now(), stopped_after_stage=stopped,
            elapsed_seconds=time.time()-started, estimated_cost_usd=(time.time()-started)/3600*manifest["cost_per_hour_usd"])
    except Exception as error:
        status.update(state="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
            elapsed_seconds=time.time()-started, estimated_cost_usd=(time.time()-started)/3600*manifest["cost_per_hour_usd"])
        raise
    finally:
        upload_json(api, repo, status, "status.json", "Update V4.2 status"); print(json.dumps(status, indent=2), flush=True)
if __name__ == "__main__": main()
