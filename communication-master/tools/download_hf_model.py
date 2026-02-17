# -*- coding: utf-8 -*-
"""
Download a Hugging Face model snapshot to a local directory.

Typical use:
  python tools/download_hf_model.py --repo_id Qwen/Qwen2.5-7B-Instruct --out models/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def _default_out_dir(repo_id: str) -> str:
    safe = repo_id.replace("/", "_")
    return os.path.join("models", safe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face repo snapshot.")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default=None, help="Local output directory (default: models/<repo_id>)")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("HF_ENDPOINT", ""),
        help="Optional endpoint/mirror (or set HF_ENDPOINT).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN", ""),
        help="Optional HF token (or set HF_TOKEN).",
    )
    parser.add_argument("--revision", default=None, help="Optional git revision / tag / commit.")
    parser.add_argument(
        "--no_hf_transfer",
        action="store_true",
        help="Disable hf_transfer acceleration.",
    )
    args = parser.parse_args()

    out_dir = args.out or _default_out_dir(args.repo_id)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    prev = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
    if args.no_hf_transfer:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        print(f">>> Downloading: {args.repo_id}")
        print(f">>> Saving to : {out_dir}")
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=out_dir,
            endpoint=args.endpoint or None,
            token=args.token or None,
            revision=args.revision,
        )
        print(">>> Done.")
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        else:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = prev


if __name__ == "__main__":
    main()

