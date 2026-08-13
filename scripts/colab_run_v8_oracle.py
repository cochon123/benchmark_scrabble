import os
import subprocess
import traceback
from pathlib import Path


os.environ["PYTHONPATH"] = "/content"
log_path = Path("/content/v8_oracle_run.log")
try:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                "python",
                "/content/scripts/evaluate_v8_oracle_interventions.py",
                "--adapter",
                "/content/artifacts/v42/hf/stage1/adapter",
                "--dataset",
                "/content/data/general_v8/oracle_interventions_46.json",
                "--output",
                "/content/v8_oracle_results.json",
                "--batch-size",
                "4",
                "--max-new-tokens",
                "96",
            ],
            check=False,
            env=os.environ,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(log_path.read_text(encoding="utf-8")[-12000:])
    if completed.returncode:
        raise SystemExit(completed.returncode)
except BaseException:
    with log_path.open("a", encoding="utf-8") as log:
        traceback.print_exc(file=log)
    print(log_path.read_text(encoding="utf-8")[-12000:])
    raise
