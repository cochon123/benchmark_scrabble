import os
import subprocess
from pathlib import Path


os.environ["PYTHONPATH"] = "/content/v8_joint"
root = Path("/content/v8_joint")
output = root / "artifacts" / "v8" / "joint"
log_path = output / "constrained_evaluation.log"
with log_path.open("a", encoding="utf-8") as log:
    correction = [
        "python", str(root / "scripts" / "evaluate_v8_joint.py"),
        "--checkpoint", str(output / "canonical-8101.pt"),
        "--positions", str(root / "data" / "general_v8_fresh" / "test_positions.json"),
        "--boards", "100", "--beam-size", "32",
        "--output", str(output / "canonical-8101-balanced-evaluation.json"),
    ]
    log.write("RUN " + " ".join(correction) + "\n"); log.flush()
    if subprocess.run(correction, cwd=root, env=os.environ, stdout=log, stderr=subprocess.STDOUT).returncode:
        raise SystemExit(1)
    for objective in ("canonical", "set-valued"):
        for seed in (8101, 8102, 8103):
            command = [
                "python", str(root / "scripts" / "evaluate_v8_joint.py"),
                "--checkpoint", str(output / f"{objective}-{seed}.pt"),
                "--positions", str(root / "data" / "general_v8_fresh" / "test_positions.json"),
                "--boards", "100", "--beam-size", "32", "--lexicon-constrained",
                "--output", str(output / f"{objective}-{seed}-constrained-evaluation.json"),
            ]
            log.write("RUN " + " ".join(command) + "\n"); log.flush()
            result = subprocess.run(command, cwd=root, env=os.environ, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise SystemExit(result.returncode)
print("V8_CONSTRAINED_COMPLETE", flush=True)
