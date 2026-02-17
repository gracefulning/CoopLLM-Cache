# -*- coding: utf-8 -*-

from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import sys
import json
import torch
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

from datasets import Dataset
from transformers import TrainerCallback, TextStreamer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# -------------------- Path setup (so imports work when run as a script) --------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 本地模块
from cache_env.unified_multi_bs_cache_env import UnifiedMultiBSCacheEnv
from cache_env.unified_data_loader import MultiUserDataLoaderZipf
from cache_env.action_validator import CacheActionValidator
from cache_env.history_rebuilder import get_sft_teaching_decision
from cache_env.prompt_utils import format_joint_state_to_llm_prompt

# ---------------- 全局配置 ----------------
MAX_SEQ_LENGTH = 2048

MODEL_PATH = os.getenv("SFT_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct"))
DATA_DIR = os.getenv("SFT_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "sft"))
OUTPUT_DIR = os.getenv("SFT_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs", "sft"))

NUM_BASE_STATIONS = 2
NUM_USERS = 20
NUM_CONTENTS = 100
CACHE_SIZES = [10] * NUM_BASE_STATIONS

NUM_EPOCHS_DATA_SFT = 3000
ZIPF_PARAM = 1.2

LORA_RANK = 64
LORA_DROPOUT = 0

SFT_BATCH_SIZE = 2
SFT_GRAD_ACCUM_STEPS = 8
SFT_LR = 1e-4
SFT_EPOCHS = 3
SFT_LOGGING_STEPS = 5
SFT_SAVE_STEPS = 25
SFT_SAVE_TOTAL_LIMIT = 3

EVAL_DATASET_SIZE = 200
SFT_DATA_SEED = 666

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- 可选：优化器回退 ----------------
try:
    import bitsandbytes as bnb  # noqa: F401
    _OPTIM = "adamw_8bit"
except Exception:
    _OPTIM = "adamw_torch"


# =================================================================
# ================== 工具：动作细节兜底方法 =======================
# =================================================================
def _get_action_details_fallback(env: UnifiedMultiBSCacheEnv, act_idx: int) -> Dict[str, Any]:
    """
    当 CacheActionValidator 缺少 get_action_details_from_index 时的兜底解码。
    约定：act_idx == 0 视为 NoOp；>0 时按 (idx-1) 解 slot/new_cid（常见编码）。
    """
    try:
        if hasattr(env, "_decode_action"):
            slot, new_cid = env._decode_action(act_idx)  # type: ignore[attr-defined]
            if act_idx == 0:
                return {"name": "CacheNoOp", "parameters": {}}
            return {"name": "CacheReplace", "parameters": {"slot_index": int(slot), "content_id": int(new_cid)}}
    except Exception:
        pass

    if act_idx == 0:
        return {"name": "CacheNoOp", "parameters": {}}
    slot = (act_idx - 1) // env.num_contents
    new_cid = (act_idx - 1) % env.num_contents
    return {"name": "CacheReplace", "parameters": {"slot_index": int(slot), "content_id": int(new_cid)}}


# =================================================================
# ================== SFT 数据生成主函数 ===========================
# =================================================================
def generate_sft_data(env: UnifiedMultiBSCacheEnv, num_samples: int, cache_validator: CacheActionValidator) -> List[Dict[str, Any]]:
    """
    生成 SFT 数据集。
    频率写入与展示遵循“单写者 + 按需加一”：
      - 每个时间步仅写一次（优先由 step() 写 t+1；仅在预热后的第一步手动补写 t）
      - prompt 仅在“当前步尚未写入”时对短/中/长的展示端 +1
    """
    sft_data = []

    print("重置环境并开始预填充...")
    env.reset(reset_caches=True, reset_freqs=True)
    env.prefill_cache_with_expert_policy(get_sft_teaching_decision, cache_validator, warmup_steps=200)
    print(f"预填充完成。当前时间步: {env.cur_epoch}")

    # 选择动作细节解码器
    if hasattr(cache_validator, "get_action_details_from_index"):
        _get_details = cache_validator.get_action_details_from_index  # type: ignore[attr-defined]
    else:
        def _get_details(act_idx: int, bs_idx: int):  # noqa: ANN001
            return _get_action_details_fallback(env, act_idx)

    pbar = tqdm(total=num_samples, desc="正在寻找已满缓存状态并生成SFT数据")
    while len(sft_data) < num_samples:
        joint_actions = None
        current_requests = None
        need_plus_one = False
        try:
            if env.cur_epoch >= env.num_epochs - 1:
                print("\n数据加载器耗尽，重置并重新预热环境。")
                env.reset(reset_caches=True, reset_freqs=True)
                env.prefill_cache_with_expert_policy(get_sft_teaching_decision, cache_validator, warmup_steps=200)
                print(f"预填充完成。当前时间步: {env.cur_epoch}")

            # === 读取当前时刻 t 的请求（先用于决策 & 展示；不要立即写入环境频率）===
            current_requests = env.requests[env.cur_epoch]
            requested_sets = env._get_requested_contents_per_bs(current_requests)

            # 使用截至 t-1 的频率做专家决策
            joint_actions = [
                get_sft_teaching_decision(env, bs_idx, requested_sets[bs_idx], cache_validator)
                for bs_idx in range(env.num_base_stations)
            ]

            # 判断缓存是否已满（全部基站）
            all_caches_are_full = all(-1 not in env.bs_caches[bs_idx] for bs_idx in range(env.num_base_stations))

            # 本步是否已写入频率（基于时间戳，而非集合相等）
            need_plus_one = (env._last_freq_epoch < env.cur_epoch)
            t_extra_per_bs = [requested_sets[b] if need_plus_one else set() for b in range(env.num_base_stations)]

            # 若未满：按需补写 t => step() 推进到 t+1（内部写 t+1）
            if not all_caches_are_full:
                if need_plus_one:
                    env._update_frequency_features(current_requests)  # 写入 t（每步集合）
                    env._last_freq_epoch = env.cur_epoch
                env.step(joint_actions)  # 写入 t+1
                continue

            # === 缓存已满：生成 SFT 样本 ===
            # 1) 构造 prompt —— 仅在需要时对短/中/长频率各 +1（按 t 步去重后的 requested_sets）
            messages = format_joint_state_to_llm_prompt(
                env,
                requested_sets,
                current_requests_sets_at_t=t_extra_per_bs
            )

            # 2) 构造 assistant 决策（只接受 CacheNoOp / CacheReplace）
            decision_lines = []
            for bs_idx, act_idx in enumerate(joint_actions):
                details = _get_details(act_idx, bs_idx)
                name = details.get("name")

                if name == "CacheNoOp":
                    decision_lines.append(f"基站{bs_idx}的决策是：不执行任何缓存操作。")
                elif name == "CacheReplace":
                    params = details.get("parameters", {})
                    slot_0 = int(params["slot_index"])
                    new_cid = int(params["content_id"])
                    old_cid = int(env.bs_caches[bs_idx][slot_0])  # 从缓存读出被驱逐内容，保证 W 一致
                    decision_lines.append(f"基站{bs_idx}的决策是：用内容{new_cid}替换槽位{slot_0 + 1}中的内容{old_cid}。")
                else:
                    raise ValueError(f"Unknown action name from get_action_details: {name}")

            assistant_content = "\n".join(decision_lines)
            messages.append({"role": "assistant", "content": assistant_content})
            sft_data.append({"conversations": messages})
            pbar.update(1)

            # 3) 按需把 t 写入频率
            if need_plus_one:
                env._update_frequency_features(current_requests)
                env._last_freq_epoch = env.cur_epoch

            # 4) 执行动作，推进到 t+1（内部写 t+1）
            env.step(joint_actions)

        except Exception as e:
            print(f"\n在数据生成步骤 {len(sft_data) + 1} 出错: {e}")
            try:
                if current_requests is not None and (env._last_freq_epoch < env.cur_epoch):
                    env._update_frequency_features(current_requests)
                    env._last_freq_epoch = env.cur_epoch
                if joint_actions is not None:
                    env.step(joint_actions)
            except Exception:
                pass
            continue
    pbar.close()
    return sft_data


# =================================================================
# ======================== 训练回调（仅观察） =====================
# =================================================================
class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, training_data_pool: List[Dict], print_every_steps: int = 100):
        self.tokenizer = tokenizer
        self.training_data_pool = training_data_pool
        self.print_every_steps = max(1, int(print_every_steps))

        # 构建更稳健的停止符列表：仅当 token 存在且非 <unk> 时加入
        raw_ids: List[int] = []
        if isinstance(tokenizer.eos_token_id, int):
            raw_ids.append(tokenizer.eos_token_id)

        for tok in ("<|endoftext|>", "<|im_end|>"):
            tid = None
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                pass
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
                raw_ids.append(tid)

        # 去重
        self.stop_token_ids: List[int] = []
        seen = set()
        for x in raw_ids:
            if x not in seen:
                self.stop_token_ids.append(x)
                seen.add(x)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 只在主进程、按频率打印
        if not state.is_world_process_zero or not self.training_data_pool:
            return
        if state.global_step % self.print_every_steps != 0:
            return

        # 获取模型（有些版本不会从 kwargs 传入）
        model = kwargs.get("model", None)
        if model is None:
            return

        sample = random.choice(self.training_data_pool)
        prompt_messages = sample["conversations"][:-1]
        ground_truth = sample["conversations"][-1].get("content", "")

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)

        print("\n" + "=" * 50 + f"  生成效果检查 @ 步骤 {state.global_step}  " + "=" * 50)
        print(f"--- 模型输入 ---\n{prompt}")
        print(f"\n--- 模型输出 (贪心) ---")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=128,
                eos_token_id=self.stop_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,      # 贪心，便于查看格式是否稳定
                streamer=streamer,
            )
        print(f"\n--- 真实答案 ---\n{ground_truth.strip()}")
        print("=" * 124 + "\n")


# =================================================================
# ======================== 训练器及主函数 =========================
# =================================================================
def main():
    # 随机种子（含 CUDA）
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_root = os.path.join(OUTPUT_DIR, f"SFT_Qwen2_Cache_{timestamp}")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(output_dir_root, exist_ok=True)

    sft_data_filename = f"sft_data_no_cot_replacement_only_v1_nbs{NUM_BASE_STATIONS}_seed{SFT_DATA_SEED}.json"
    sft_generated_json_file = os.path.join(DATA_DIR, sft_data_filename)

    if not os.path.exists(sft_generated_json_file):
        print("SFT数据文件未找到，正在生成...")
        data_loader = MultiUserDataLoaderZipf(
            NUM_CONTENTS, NUM_EPOCHS_DATA_SFT + 500, NUM_USERS, ZIPF_PARAM, seed=SFT_DATA_SEED
        )
        env = UnifiedMultiBSCacheEnv(data_loader, NUM_BASE_STATIONS, CACHE_SIZES, NUM_USERS, NUM_CONTENTS)
        cache_validator = CacheActionValidator(NUM_CONTENTS, CACHE_SIZES)
        sft_data_list = generate_sft_data(env, NUM_EPOCHS_DATA_SFT, cache_validator)
        with open(sft_generated_json_file, "w", encoding="utf-8") as f:
            json.dump(sft_data_list, f, ensure_ascii=False, indent=2)
        print(f"SFT数据已生成: {sft_generated_json_file}")
    else:
        print(f"正在加载SFT数据: {sft_generated_json_file}")
        with open(sft_generated_json_file, "r", encoding="utf-8") as f:
            sft_data_list = json.load(f)

    if not sft_data_list:
        print("错误：SFT数据为空，无法进行训练。请检查数据生成过程。")
        return

    random.shuffle(sft_data_list)
    train_data_raw = sft_data_list[EVAL_DATASET_SIZE:] if len(sft_data_list) > EVAL_DATASET_SIZE else sft_data_list
    if not train_data_raw:
        print("错误：划分后训练集为空。")
        return

    # ====== 1) 构建 HF 数据集 ======
    raw_ds = Dataset.from_list(train_data_raw)

    # ====== 2) 预加载模型与分词器（Unsloth） ======
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    # pad 对齐
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # QLoRA（Unsloth）
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ====== 3) 用 chat_template 生成“整段文本”并过滤超长 ======
    def add_text(example):
        conv = example["conversations"]
        text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = raw_ds.map(add_text, remove_columns=["conversations"])

    # 统计长度并过滤
    def count_len(example):
        ids = tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        example["token_length"] = len(ids)
        return example

    ds = ds.map(count_len)
    ds = ds.filter(lambda ex: ex["token_length"] <= MAX_SEQ_LENGTH)
    ds = ds.remove_columns(["token_length"])

    # ====== 4) 定义“只训回复”的 collator ======
    RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
    resp_tmpl_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=resp_tmpl_ids,
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
    )

    print(f"训练集大小: {len(ds)}")

    # ====== 5) Trainer ======
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=data_collator,
        args=SFTConfig(
            packing=False,
            per_device_train_batch_size=SFT_BATCH_SIZE,
            gradient_accumulation_steps=SFT_GRAD_ACCUM_STEPS,
            max_grad_norm=1.0,
            warmup_ratio=0.05,
            num_train_epochs=SFT_EPOCHS,
            learning_rate=SFT_LR,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=SFT_LOGGING_STEPS,
            optim=_OPTIM,
            seed=42,
            output_dir=output_dir_root,
            save_strategy="steps",
            save_steps=SFT_SAVE_STEPS,
            save_total_limit=SFT_SAVE_TOTAL_LIMIT,
            report_to="none",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
        ),
        callbacks=[GenerationCallback(tokenizer, train_data_raw, print_every_steps=5)],
    )

    print("\n开始SFT训练（官方路径，completion-only 由 collator 处理）...")
    trainer.train()
    print("\nSFT训练完成。")

    final_model_save_path = os.path.join(output_dir_root, "final_sft_checkpoint")
    trainer.save_model(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    print(f"最终SFT模型已保存至: {final_model_save_path}")


if __name__ == "__main__":
    main()
