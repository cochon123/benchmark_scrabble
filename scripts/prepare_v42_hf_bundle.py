#!/usr/bin/env python3
from __future__ import annotations
import gzip, hashlib, json, tarfile
from pathlib import Path
from huggingface_hub import HfApi

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def archive(output, entries):
    with Path(output).open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as target:
                for path, name in sorted(entries, key=lambda x:x[1]):
                    info=target.gettarinfo(str(path),arcname=name); info.uid=info.gid=0; info.uname=info.gname=""; info.mtime=0
                    with path.open("rb") as handle: target.addfile(info,handle)
def main():
    root=Path(__file__).resolve().parents[1]; output=root/"artifacts/v42_hf_bundle"; output.mkdir(parents=True,exist_ok=True)
    source=[(root/"pyproject.toml","pyproject.toml"),(root/"data/lexicon/ENABLE.txt","data/lexicon/ENABLE.txt")]
    source += [(p,str(p.relative_to(root))) for folder in ("scrabble_bench","scripts") for p in (root/folder).rglob("*.py")]
    archive(output/"source.tar.gz",source)
    data=root/"data/general_v42"; archive(output/"data.tar.gz",[(p,str(p.relative_to(root))) for p in data.rglob("*") if p.is_file()])
    manifest={"version":"v4.2-job-1","output_repo":"Cochon123/Qwen3-4B-Scrabble-General-v4.2",
      "cost_per_hour_usd":1.8,"timeout_hours":1.5,"max_cost_usd":2.7,
      "stages":[
        {"name":"stage1","file":"stage1.jsonl","steps":500,"lr":3e-5,"gate_file":"localization_gate.json","gate":[["start-choice","success_pct",50],["start-location","success_pct",25]]},
        {"name":"stage2","file":"stage2.jsonl","steps":500,"lr":2e-5,"gate_file":"word_move_gate.json","gate":[["word-direction-move","legal_pct",20]]},
        {"name":"stage3","file":"stage3.jsonl","steps":400,"lr":1e-5,"gate_file":"free_gate.json","gate":[]}]}
    manifest["sha256"]={n:sha256(output/n) for n in ("source.tar.gz","data.tar.gz")}
    (output/"job_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    api=HfApi(); input_repo="Cochon123/scrabble-v42-localization"; api.create_repo(input_repo,repo_type="dataset",private=True,exist_ok=True)
    api.create_repo(manifest["output_repo"],private=True,exist_ok=True)
    api.upload_folder(folder_path=str(output),repo_id=input_repo,repo_type="dataset",commit_message="Upload audited V4.2 bundle")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
