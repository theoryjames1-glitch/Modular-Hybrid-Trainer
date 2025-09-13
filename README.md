
# 1) Add guarded imports to `rewards.py`

Place near the other optional imports:

```python
# BERTScore (optional)
try:
    from bert_score import score as bertscore_score
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False

# Transformers LM for Perplexity (we assume transformers is present in your stack)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TXF = True
except Exception:
    _HAS_TXF = False
```

---

# 2) Add reward classes

Put these after the Toxicity class (or wherever you grouped optional metrics):

```python
# -----------------------------
# BERTScore (F1) in [0,1]
# -----------------------------
class BERTScore(Reward):
    """
    Returns BERTScore F1 ∈ [0,1].
    You can set model_type (e.g., 'microsoft/deberta-xlarge-mnli') and language.
    """
    def __init__(self, model_type: str | None = None, lang: str | None = None,
                 rescale_with_baseline: bool = False, device: str | None = None):
        if not _HAS_BERTSCORE:
            raise ImportError("bert-score not installed. pip install bert-score")
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = bool(rescale_with_baseline)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def __call__(self, response, target, **_):
        hyp = [response or ""]
        ref = [target or ""]
        P, R, F1 = bertscore_score(
            hyp, ref,
            model_type=self.model_type,
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline,
            device=self.device,
            verbose=False,
        )
        return float(F1.mean().item())  # already in [0,1]


# -----------------------------
# Perplexity-based reward
# -----------------------------
class LMPerplexity(Reward):
    """
    Reward based on language-model perplexity of the RESPONSE (lower PPL => higher reward).
    Default reward = exp(-loss) = 1 / ppl ∈ (0,1]; can choose 'inverse' or raw 'neg_loss'.
    - model_name: LM used to score (e.g., 'gpt2', 'gpt2-medium', 'EleutherAI/pythia-410m')
    - stride: evaluate long text in sliding windows to avoid OOM.
    """
    _CACHE: dict[str, tuple[AutoModelForCausalLM, AutoTokenizer, str]] = {}

    def __init__(self, model_name: str = "gpt2", device: str | None = None,
                 stride: int = 512, max_length: int = 1024,
                 reward_mode: str = "exp_neg_loss",  # 'exp_neg_loss' (1/ppl), 'inverse', or 'neg_loss'
                 add_special_tokens: bool = False):
        if not _HAS_TXF:
            raise ImportError("transformers not available for LMPerplexity")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stride = int(stride)
        self.max_length = int(max_length)
        self.reward_mode = str(reward_mode).lower()
        self.add_special_tokens = bool(add_special_tokens)

        if model_name not in LMPerplexity._CACHE:
            tok = AutoTokenizer.from_pretrained(model_name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            mdl.eval()
            LMPerplexity._CACHE[model_name] = (mdl, tok, self.device)

        self.model, self.tokenizer, _ = LMPerplexity._CACHE[model_name]

    @torch.inference_mode()
    def __call__(self, response, target=None, **_):
        text = (response or "").strip()
        if not text:
            # Empty strings are "easy"; give neutral reward
            return 0.5

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
            truncation=False
        )
        input_ids = enc["input_ids"][0]

        nll, count = 0.0, 0
        # sliding window to handle long inputs
        for i in range(0, input_ids.size(0), self.stride):
            begin = max(0, i + input_ids.size(0) - self.max_length)  # keep tail up to max_length
            end = min(i + self.stride, input_ids.size(0))
            trg = input_ids[begin:end].unsqueeze(0).to(self.device)
            # labels are same as input; shift handled by model loss
            out = self.model(trg, labels=trg)
            loss = float(out.loss.item())  # average NLL over tokens in this chunk
            n_tokens = trg.size(1)
            nll += loss * n_tokens
            count += n_tokens

            if end == input_ids.size(0):
                break

        mean_nll = nll / max(1, count)  # average negative log-likelihood per token
        # ppl = exp(mean_nll)
        if self.reward_mode in ("exp_neg_loss", "inverse"):
            # exp(-loss) == 1 / exp(loss) == 1 / ppl ∈ (0,1]
            reward = math.exp(-mean_nll)
        elif self.reward_mode == "neg_loss":
            # raw negative loss (higher is better, unbounded); map to (0,1) softly
            reward = 1.0 - (1.0 - math.exp(-mean_nll))  # equivalent but keeps (0,1]
        else:
            # default to exp_neg_loss
            reward = math.exp(-mean_nll)

        # clamp numerical issues
        return max(0.0, min(1.0, float(reward)))
```

---

# 3) Extend the factory mapping

Add these cases in your `_build_leaf(spec)`:

```python
    if t in ("bertscore", "bert_score", "bert-score"):
        return BERTScore(
            model_type=spec.get("model_type"),
            lang=spec.get("lang"),
            rescale_with_baseline=bool(spec.get("rescale_with_baseline", False)),
            device=spec.get("device"),
        )
    if t in ("perplexity", "ppl"):
        return LMPerplexity(
            model_name=spec.get("model", "gpt2"),
            device=spec.get("device"),
            stride=int(spec.get("stride", 512)),
            max_length=int(spec.get("max_length", 1024)),
            reward_mode=spec.get("reward_mode", "exp_neg_loss"),
            add_special_tokens=bool(spec.get("add_special_tokens", False)),
        )
```

That’s it — your reward factory now supports BERTScore and Perplexity.

---

## 4) Example JSON configs

**A) BERTScore-only (DeBERTa model, baseline rescale):**

```json
{
  "REWARD": {
    "type": "bert_score",
    "model_type": "microsoft/deberta-xlarge-mnli",
    "rescale_with_baseline": true,
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

**B) Perplexity (reward = 1/ppl using GPT-2 scorer):**

```json
{
  "REWARD": {
    "type": "perplexity",
    "model": "gpt2",
    "stride": 512,
    "max_length": 1024,
    "reward_mode": "exp_neg_loss",   // 1/ppl in (0,1]
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

**C) Weighted mix: BERTScore 0.6 + ROUGE-L 0.3 + Low-Toxicity 0.1**

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "bert_score", "model_type": "microsoft/deberta-xlarge-mnli", "rescale_with_baseline": true, "weight": 0.6},
      {"type": "rouge_l", "use_stemmer": true, "weight": 0.3},
      {"type": "toxicity", "invert": true, "weight": 0.1}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

**D) Semantics + Fluency (1/ppl) + EMA normalization for PPO stability**

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "cosine_sts", "weight": 0.5},
      {"type": "perplexity", "model": "gpt2-medium", "weight": 0.5}
    ],
    "postprocess": [
      {"type": "ema_normalize", "decay": 0.995}
    ]
  }
}
```

---

## 5) Install notes

* BERTScore: `pip install bert-score`
* ROUGE: `pip install rouge-score`
* BLEU: `pip install sacrebleu`
* Toxicity: `pip install detoxify` (and appropriate PyTorch for your CUDA/CPU)

**Performance tips**

* BERTScore with large models (e.g., DeBERTa-XL) is heavy; consider batching or a smaller model if throughput matters.
* Perplexity evaluation uses a sliding window; tune `stride`/`max_length` based on VRAM.
* If you’re mixing many rewards, consider wrapping with `EMANormalize` to stabilize PPO updates.
