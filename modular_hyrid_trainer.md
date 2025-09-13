awesome — here’s a **modular Hybrid Trainer** that lets you run **SFT**, **PPO**, or a **hybrid (interleaved)** loop from one JSON config. It plugs into your existing `rewards.py` factory (the one we built), uses one model (`AutoModelForCausalLMWithValueHead`) and alternates SFT updates (CE loss on the policy head) with PPO updates (policy/value). It’s lean, explicit, and easy to tweak.

Run:

```
python hybrid_trainer.py your_config.json
```

---

### `hybrid_trainer.py`

````python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, random, gc
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    StoppingCriteria, StoppingCriteriaList,
    get_scheduler as hf_get_scheduler,
    set_seed,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# ==== optional external reward factory (from prior step) ====
# pip install sentence-transformers detoxify sacrebleu rouge-score bert-score
from rewards import make_reward  # <-- our modular reward factory

# ----------------------------
# Utils
# ----------------------------
def get_os_seed(nbytes: int = 4) -> int:
    return int.from_bytes(os.urandom(nbytes), byteorder="big")

def str2bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")

def ensure_tokenizer(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def pick_dtype(bf16: bool, fp16: bool):
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None

# ----------------------------
# Stopping criteria (stop on EOS)
# ----------------------------
class StopOnEOSToken(StoppingCriteria):
    def __init__(self, eos_token_id: int):
        super().__init__()
        self.eos_token_id = eos_token_id
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_) -> bool:
        return input_ids[0, -1].item() == self.eos_token_id

# ----------------------------
# Data
# ----------------------------
class JsonListDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], select_output: str = "output"):
        self.items = items
        self.select_output = select_output
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

def build_prompt_and_target(item: dict, output_key: str = "output") -> Tuple[str, str]:
    prompt = '### Prompt:\n\n```'
    for k, v in item.items():
        if k == output_key or v is None:
            continue
        if isinstance(v, list):
            prompt += f"{k}: {','.join(map(str, v))}\n"
        else:
            prompt += f"{k}: {str(v)}\n"
    prompt = prompt.strip() + "```\n\n### Response:\n\n"
    out = item.get(output_key, "")
    if isinstance(out, list):
        out = out[0] if out else ""
    target = str(out)
    return prompt, target

# ----------------------------
# SFT step (cross-entropy on response tokens only)
# ----------------------------
def sft_step(model, tokenizer, batch_items, optimizer, maxseq: int, device: str):
    # Build batch texts and label masks
    inputs_list, labels_list = [], []
    eos = tokenizer.eos_token
    for item in batch_items:
        prompt, target = build_prompt_and_target(item, output_key=item.get("_output_key", "output"))
        text = f"{prompt}{target}{eos}"
        # Encode full text (prompt + target + eos)
        enc = tokenizer(text, truncation=True, padding=False, max_length=maxseq, add_special_tokens=False)
        input_ids = enc["input_ids"]
        # Encode prompt alone to mask it out
        prompt_ids = tokenizer(prompt, truncation=True, padding=False, max_length=maxseq, add_special_tokens=False)["input_ids"]
        labels = input_ids.copy()
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)  # mask prompt tokens
        inputs_list.append(torch.tensor(input_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    # Pad to batch
    input_ids = torch.nn.utils.rnn.pad_sequence(inputs_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels    = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    input_ids, labels = input_ids.to(device), labels.to(device)

    outputs = model.pretrained_model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.pretrained_model.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.item())

# ----------------------------
# PPO collect & update
# ----------------------------
@torch.no_grad()
def ppo_collect_batch(model, tokenizer, items, maxseq, max_new_tokens, gen_cfg, stopping, device: str):
    batch_queries, batch_responses, texts = [], [], []
    eos_id = tokenizer.eos_token_id
    for it in items:
        prompt, _ = build_prompt_and_target(it, output_key=it.get("_output_key", "output"))
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=maxseq)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        response_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=gen_cfg.get("top_p", 0.9),
            temperature=gen_cfg.get("temperature", 0.7),
            no_repeat_ngram_size=gen_cfg.get("no_repeat_ngram_size", 3),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.15),
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            use_cache=False,
            stopping_criteria=stopping,
        )
        batch_queries.append(inputs["input_ids"][0].detach().cpu())
        batch_responses.append(response_ids[0].detach().cpu())
        txt = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        if txt.startswith(prompt):
            txt = txt[len(prompt):].strip()
        texts.append((prompt, txt))
    return batch_queries, batch_responses, texts

# ----------------------------
# Hybrid Trainer
# ----------------------------
def train_hybrid(cfg_path: str):
    # ----- Config -----
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    seed = cfg.get("SEED")
    if seed is None:
        seed = get_os_seed()
    set_seed(int(seed))

    MODEL         = cfg["MODEL"]
    TRAIN_FILE    = cfg["TRAIN_FILE"]
    OUTPUT_DIR    = cfg["OUTPUT_DIR"]
    MODE          = str(cfg.get("MODE", "hybrid")).lower()  # 'sft' | 'ppo' | 'hybrid'

    # data settings
    SELECT_OUTPUT = cfg.get("SELECT_OUTPUT", "output")
    PERCENT       = int(cfg.get("PERCENT", 100))
    NUM_SAMPLES   = int(cfg.get("NUM_SAMPLES", 0))
    SHUFFLE       = cfg.get("SHUFFLE")

    # sft
    SFT_BATCH     = int(cfg.get("SFT_BATCH", 4))
    LR_SFT        = float(cfg.get("LR_SFT", 5e-5))

    # ppo
    PPO_BATCH     = int(cfg.get("PPO_BATCH", 4))       # samples per PPO update
    PPO_MINI      = int(cfg.get("PPO_MINI", PPO_BATCH))
    LR_PPO        = float(cfg.get("LR_PPO", 1e-5))
    TARGET_KL     = float(cfg.get("TARGET_KL", 0.3))

    # interleave
    CYCLE_SFT_STEPS = int(cfg.get("CYCLE_SFT_STEPS", 1))   # SFT steps per cycle
    CYCLE_PPO_UPDS  = int(cfg.get("CYCLE_PPO_UPDATES", 1)) # PPO updates per cycle
    EPOCHS          = int(cfg.get("EPOCHS", 1))

    # tokens
    MAXSEQ        = int(cfg.get("MAXSEQ", 1024))
    MAX_NEW       = int(cfg.get("MAX_NEW_TOKENS", 128))

    # dtype
    BF16          = str2bool(cfg.get("BF16", "false"))
    FP16          = str2bool(cfg.get("FP16", "false"))
    torch_dtype   = pick_dtype(BF16, FP16)

    # generation config (ppo collection)
    GEN_CFG = {
        "top_p": float(cfg.get("TOP_P", 0.9)),
        "temperature": float(cfg.get("TEMPERATURE", 0.7)),
        "no_repeat_ngram_size": int(cfg.get("NO_REPEAT_NGRAM_SIZE", 3)),
        "repetition_penalty": float(cfg.get("REPETITION_PENALTY", 1.15)),
    }

    # reward spec (from rewards.py)
    REWARD_SPEC = cfg.get("REWARD", {"type": "cosine_sts"})

    # ----- Data -----
    if SHUFFLE:
        os.system(f"python {SHUFFLE}")
    with open(TRAIN_FILE, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("TRAIN_FILE must be a JSON list of samples")
    random.shuffle(data)
    if NUM_SAMPLES <= 0:
        NUM_SAMPLES = int(len(data) * (PERCENT / 100.0))
    data = data[:NUM_SAMPLES]
    # store output key inside items (for custom keys)
    for it in data:
        it["_output_key"] = SELECT_OUTPUT

    ds = JsonListDataset(data, select_output=SELECT_OUTPUT)
    # separate loaders for SFT and PPO to keep batch sizes independent
    sft_loader = DataLoader(ds, batch_size=SFT_BATCH, shuffle=True, drop_last=False)
    ppo_loader = DataLoader(ds, batch_size=PPO_BATCH, shuffle=True, drop_last=False)

    # ----- Tokenizer / Model with value head -----
    base_model = OUTPUT_DIR if os.path.isdir(OUTPUT_DIR) else MODEL
    print("BASE_MODEL:", base_model)

    tokenizer = ensure_tokenizer(AutoTokenizer.from_pretrained(base_model))
    eos_id = tokenizer.eos_token_id
    stopping = StoppingCriteriaList([StopOnEOSToken(eos_id)])

    device_map = "auto"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Optimizer for SFT path (policy-only) -----
    opt_sft = torch.optim.AdamW(model.pretrained_model.parameters(), lr=LR_SFT)
    # (You can wire your make_scheduler() here if desired.)

    # ----- PPO path -----
    ppo_cfg = PPOConfig(
        model_name=base_model,
        learning_rate=LR_PPO,
        batch_size=PPO_BATCH,
        mini_batch_size=PPO_MINI,
        kl_penalty="kl",
        target_kl=TARGET_KL,
        seed=int(seed),
        remove_unused_columns=True,
        log_with=None,  # 'wandb'/'tensorboard' if you want
        accelerator_kwargs={"log_with": None},
    )
    ppo_trainer = PPOTrainer(config=ppo_cfg, model=model, tokenizer=tokenizer)

    # Reward function (from rewards factory)
    reward_fn = make_reward(REWARD_SPEC)

    print("✅ Hybrid trainer initialized")
    print(f"Mode: {MODE}  |  SFT_BATCH={SFT_BATCH}  PPO_BATCH={PPO_BATCH}  Cycles: SFT {CYCLE_SFT_STEPS} / PPO {CYCLE_PPO_UPDS}")

    # ----- Training -----
    global_step = 0
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        # fresh shuffles each epoch
        sft_iter = iter(DataLoader(ds, batch_size=SFT_BATCH, shuffle=True))
        ppo_iter = iter(DataLoader(ds, batch_size=PPO_BATCH, shuffle=True))

        keep_going = True
        while keep_going:
            # ---- SFT phase ----
            if MODE in ("sft", "hybrid"):
                for _ in range(CYCLE_SFT_STEPS):
                    try:
                        batch_items = next(sft_iter)
                    except StopIteration:
                        keep_going = False
                        break
                    loss = sft_step(model, tokenizer, batch_items, opt_sft, MAXSEQ, device)
                    global_step += 1
                    if global_step % 10 == 0:
                        print(f"[SFT] step={global_step} loss={loss:.4f}")

            # ---- PPO phase ----
            if MODE in ("ppo", "hybrid"):
                for _ in range(CYCLE_PPO_UPDS):
                    try:
                        ppo_items = next(ppo_iter)
                    except StopIteration:
                        keep_going = False
                        break
                    # collect responses
                    q, r, texts = ppo_collect_batch(model, tokenizer, ppo_items, MAXSEQ, MAX_NEW, GEN_CFG, stopping, device)
                    # compute rewards
                    rewards = []
                    for (prompt, resp), item in zip(texts, ppo_items):
                        _, target = build_prompt_and_target(item, output_key=item.get("_output_key", "output"))
                        rew = reward_fn(resp, target, prompt=prompt, item=item)
                        rewards.append(float(rew))
                    # update with PPO
                    ppo_trainer.step(q, r, rewards)
                    global_step += 1
                    print(f"[PPO]  step={global_step} reward_mean={sum(rewards)/max(1,len(rewards)):.4f}")

            # Exit if both iterators are exhausted
            if MODE == "sft" and keep_going is False: break
            if MODE == "ppo" and keep_going is False: break
            if MODE == "hybrid" and keep_going is False: break

            # Housekeeping
            if str2bool(cfg.get("CLEANSE", "true")):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

    print("\n✅ Training complete.")

    # ----- Save models -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # A) PPO bundle (resume RL later)
    ppo_dir = os.path.join(OUTPUT_DIR, "ppo_full")
    os.makedirs(ppo_dir, exist_ok=True)
    ppo_trainer.save_pretrained(ppo_dir)
    tokenizer.save_pretrained(ppo_dir)
    print(f"💾 saved PPO (policy + value head) -> {ppo_dir}")

    # B) Inference-only LM
    lm_dir = os.path.join(OUTPUT_DIR, "inference_lm")
    os.makedirs(lm_dir, exist_ok=True)
    model.pretrained_model.save_pretrained(lm_dir)
    tokenizer.save_pretrained(lm_dir)
    print(f"💾 saved inference model (CausalLM) -> {lm_dir}")


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hybrid_trainer.py <config.json>")
        sys.exit(1)
    train_hybrid(sys.argv[1])
````

---

### Minimal config examples

**(A) SFT only**

```json
{
  "MODE": "sft",
  "MODEL": "gpt2",
  "TRAIN_FILE": "train.json",
  "OUTPUT_DIR": "./out_sft",
  "SFT_BATCH": 8,
  "LR_SFT": 5e-5,
  "EPOCHS": 1,
  "MAXSEQ": 1024
}
```

**(B) PPO only (cosine STS reward)**

```json
{
  "MODE": "ppo",
  "MODEL": "gpt2",
  "TRAIN_FILE": "train.json",
  "OUTPUT_DIR": "./out_ppo",
  "PPO_BATCH": 4,
  "PPO_MINI": 4,
  "LR_PPO": 1e-5,
  "TARGET_KL": 0.3,
  "EPOCHS": 1,
  "MAXSEQ": 1024,
  "MAX_NEW_TOKENS": 128,
  "REWARD": {"type": "cosine_sts"}
}
```

**(C) Hybrid (1 SFT step, then 1 PPO update, repeat)**

```json
{
  "MODE": "hybrid",
  "MODEL": "gpt2",
  "TRAIN_FILE": "train.json",
  "OUTPUT_DIR": "./out_hybrid",
  "SFT_BATCH": 8,
  "LR_SFT": 5e-5,
  "PPO_BATCH": 4,
  "PPO_MINI": 4,
  "LR_PPO": 1e-5,
  "TARGET_KL": 0.3,
  "CYCLE_SFT_STEPS": 1,
  "CYCLE_PPO_UPDATES": 1,
  "EPOCHS": 1,
  "MAXSEQ": 1024,
  "MAX_NEW_TOKENS": 128,
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "cosine_sts", "weight": 0.6},
      {"type": "perplexity", "model": "gpt2", "weight": 0.4}
    ],
    "postprocess": [{"type": "clip", "min": 0, "max": 1}]
  }
}
```

---

### Notes / gotchas (quick)

* The SFT step updates **policy weights only** (`model.pretrained_model.parameters()`), not the value head. PPO updates both policy + value head.
* Alternating optimizers like this is a practical, common pattern for hybrid online finetuning.
* If you want **exact resume**, you can add the same schedule metadata trick we used earlier and periodic `ppo_trainer.save_pretrained()` checkpoints by step.
* To make SL loss mask only the response, we mask prompt tokens with `-100` (already done).

If you want this to use your **optimizer/scheduler factories** for the SFT path instead of the simple AdamW, I can wire those in too.
