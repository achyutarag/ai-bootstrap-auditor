"""
Downloads per-question evaluation results for Mistral-7B-Instruct v0.1 and
v0.2 from the archived (public, ungated) Open LLM Leaderboard v1 details on
Hugging Face. ~190 MB total; files land in data/raw/ (git-ignored).

    python scripts/fetch_real_data.py

Tasks: ARC-Challenge, HellaSwag, MMLU (57 subjects), TruthfulQA-MC, Winogrande,
GSM8K. DROP is skipped (v0.2 was never evaluated on it). If a task was run
more than once, the latest run is used.
"""
import os
import re

import httpx
from huggingface_hub import HfApi, hf_hub_url

REPO_TEMPLATE = "open-llm-leaderboard-old/details_mistralai__Mistral-7B-Instruct-{version}"
VERSIONS = ("v0.1", "v0.2")
TASK_PATTERN = re.compile(r"details_harness\|(arc:challenge|hellaswag|hendrycksTest-[\w_]+|truthfulqa:mc|"
                          r"winogrande|gsm8k)\|\d+_(?P<ts>[\dT\-.]+)\.parquet$")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latest_task_files(repo_id):
    latest = {}
    for path in HfApi().list_repo_files(repo_id, repo_type="dataset"):
        m = TASK_PATTERN.search(path)
        if m and (m.group(1) not in latest or m.group("ts") > latest[m.group(1)][0]):
            latest[m.group(1)] = (m.group("ts"), path)
    return {task: path for task, (_, path) in latest.items()}


if __name__ == "__main__":
    for version in VERSIONS:
        repo_id = REPO_TEMPLATE.format(version=version)
        files = latest_task_files(repo_id)
        out_dir = os.path.join(ROOT, "data", "raw", f"mistral-{version}")
        print(f"{repo_id}: {len(files)} task files -> {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        for task, path in sorted(files.items()):
            # Repo file names contain '|' and ':', which Windows forbids, so
            # stream each file to a sanitized name: <task>.parquet
            dest = os.path.join(out_dir, re.sub(r"[^\w\-]", "_", task) + ".parquet")
            if os.path.exists(dest):
                continue
            with httpx.stream("GET", hf_hub_url(repo_id, path, repo_type="dataset"),
                              follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                with open(dest + ".part", "wb") as fh:
                    for chunk in response.iter_bytes():
                        fh.write(chunk)
            os.replace(dest + ".part", dest)
        print("  done")
