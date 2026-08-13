import os
import subprocess
from pathlib import Path


os.environ["PYTHONPATH"] = "/content/v8_joint"
root = Path("/content/v8_joint")
output = root / "artifacts" / "v8" / "joint"
output.mkdir(parents=True, exist_ok=True)
log_path = output / "colab_run.log"

commands = []
for objective in ("canonical", "set-valued"):
    for seed in (8101, 8102, 8103):
        checkpoint = output / f"{objective}-{seed}.pt"
        commands.append(
            [
                "python", str(root / "scripts" / "train_v8_joint.py"),
                "--train", str(root / "data" / "general_v6_positions" / "train_positions.json"),
                "--validation", str(root / "data" / "general_v6_positions" / "validation_positions.json"),
                "--objective", objective, "--seed", str(seed), "--steps", "1200",
                "--batch-size", "48", "--output", str(checkpoint),
            ]
        )
        commands.append(
            [
                "python", str(root / "scripts" / "evaluate_v8_joint.py"),
                "--checkpoint", str(checkpoint),
                "--positions", str(root / "data" / "general_v8_fresh" / "test_positions.json"),
                "--boards", "100", "--beam-size", "32",
                "--output", str(output / f"{objective}-{seed}-evaluation.json"),
            ]
        )

with log_path.open("a", encoding="utf-8") as log:
    for command in commands:
        print("RUN", " ".join(command), flush=True)
        log.write("RUN " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=root, env=os.environ, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            log.write(f"FAILED {result.returncode}\n")
            log.flush()
            raise SystemExit(result.returncode)
print("V8_JOINT_COMPLETE", flush=True)
