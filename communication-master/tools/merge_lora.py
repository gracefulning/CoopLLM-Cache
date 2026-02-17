# -*- coding: utf-8 -*-
"""
Merge a LoRA adapter into a base model and save a merged Hugging Face model.

Typical use:
  python tools/merge_lora.py --base_model models/Qwen2.5-7B-Instruct --adapter outputs/sft/.../final_sft_checkpoint --out models/merge7B_exbert
"""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


def _parse_dtype(dtype: str) -> torch.dtype:
    d = dtype.strip().lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16"}:
        return torch.float16
    if d in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base model path or repo id (Hugging Face format).",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="LoRA adapter path (a PEFT adapter folder).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory to save the merged model.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype when loading the base model.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit. Off by default for safer merging.",
    )
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)

    print(f">>> Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        dtype=dtype,
        load_in_4bit=bool(args.load_in_4bit),
    )

    print(f">>> Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print(">>> Merging adapter weights into base model...")
    model = model.merge_and_unload()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f">>> Saving merged model to: {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(">>> Done.")


if __name__ == "__main__":
    main()

