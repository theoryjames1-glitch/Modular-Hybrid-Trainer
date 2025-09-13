# 1) Drop-in patch for `rewards.py`

Add these imports near the top (right under the existing ones):

```python
# Optional metrics (guarded imports)
try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except Exception:
    _HAS_ROUGE = False

try:
    # Detoxify uses a small transformer to score toxicity locally (no API needed)
    from detoxify import Detoxify
    _HAS_DETOXIFY = True
except Exception:
    _HAS_DETOXIFY = False
```

Add these classes (e.g., after `CosineSTS`):

```python
# -----------------------------
# BLEU (sacrebleu)
# -----------------------------
class SacreBLEU(Reward):
    """
    Sentence-level BLEU via sacreBLEU.
    Returns BLEU in [0, 1] by dividing sacrebleu score (0..100) by 100.
    """
    def __init__(self, smooth_method: str = "exp", lowercase: bool = True, tokenize: str = "13a"):
        if not _HAS_SACREBLEU:
            raise ImportError("sacrebleu not installed. pip install sacrebleu")
        self.smooth_method = smooth_method
        self.lowercase = lowercase
        self.tokenize = tokenize

    def __call__(self, response, target, **_):
        hyp = response or ""
        ref = target or ""
        if self.lowercase:
            hyp, ref = hyp.lower(), ref.lower()
        # sacrebleu expects lists of refs; single ref -> [ref]
        bleu = sacrebleu.corpus_bleu(
            [hyp], [[ref]],
            smooth_method=self.smooth_method,
            lowercase=self.lowercase,
            tokenize=self.tokenize
        ).score
        return max(0.0, min(1.0, bleu / 100.0))


# -----------------------------
# ROUGE-L (rouge_score)
# -----------------------------
class RougeL(Reward):
    """
    ROUGE-L F1 via rouge_score. Returns value in [0,1].
    """
    def __init__(self, use_stemmer: bool = True, lowercase: bool = True):
        if not _HAS_ROUGE:
            raise ImportError("rouge-score not installed. pip install rouge-score")
        self.lowercase = lowercase
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)

    def __call__(self, response, target, **_):
        hyp = response or ""
        ref = target or ""
        if self.lowercase:
            hyp, ref = hyp.lower(), ref.lower()
        scores = self.scorer.score(ref, hyp)  # (reference, hypothesis)
        val = scores["rougeL"].fmeasure  # ∈ [0,1]
        return max(0.0, min(1.0, float(val)))


# -----------------------------
# Toxicity (Detoxify)
# -----------------------------
class Toxicity(Reward):
    """
    Detoxify returns toxicity probabilities in [0,1].
    By default we return (1 - toxicity) so 'less toxic' => higher reward.
    Set invert=False to reward toxicity directly (not recommended).
    """
    _MODEL_CACHE: dict[str, Detoxify] = {}

    def __init__(self, model: str = "original", device: str | None = None, invert: bool = True):
        if not _HAS_DETOXIFY:
            raise ImportError("detoxify not installed. pip install detoxify")
        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        key = f"{model}|{self.device}"
        if key not in Toxicity._MODEL_CACHE:
            Toxicity._MODEL_CACHE[key] = Detoxify(model=model, device=self.device)
        self.detox = Toxicity._MODEL_CACHE[key]
        self.invert = invert

    @torch.inference_mode()
    def __call__(self, response, target, **_):
        txt = (response or "").strip()
        if not txt:
            return 1.0 if self.invert else 0.0
        # Detoxify returns multiple heads; we use 'toxicity'
        scores = self.detox.predict(txt)
        tox = float(scores.get("toxicity", 0.0))
        return max(0.0, min(1.0, 1.0 - tox if self.invert else tox))
```

Extend your factory mapping in `_build_leaf`:

```python
def _build_leaf(spec: Dict[str, Any]) -> Reward:
    t = spec.get("type", "").lower()
    if t == "exact_match":
        return ExactMatch()
    if t == "keyword_count":
        return KeywordCount(keywords=spec.get("keywords", []), normalize=bool(spec.get("normalize", True)))
    if t == "regex_presence":
        return RegexPresence(pattern=spec["pattern"], flags=re.IGNORECASE if spec.get("ignore_case", True) else 0)
    if t == "jaccard":
        return JaccardUnigram(lowercase=bool(spec.get("lowercase", True)))
    if t == "length_window":
        return LengthWindow(min_tokens=int(spec.get("min", 30)), max_tokens=int(spec.get("max", 200)))
    if t in ("cosine_sts", "sts"):
        return CosineSTS(model_name=spec.get("model", "all-MiniLM-L6-v2"), device=spec.get("device"))
    if t == "bleu":
        return SacreBLEU(
            smooth_method=spec.get("smooth_method", "exp"),
            lowercase=bool(spec.get("lowercase", True)),
            tokenize=spec.get("tokenize", "13a"),
        )
    if t in ("rouge_l", "rouge-l", "rougel"):
        return RougeL(use_stemmer=bool(spec.get("use_stemmer", True)),
                      lowercase=bool(spec.get("lowercase", True)))
    if t in ("toxicity", "detoxify"):
        return Toxicity(
            model=spec.get("model", "original"),
            device=spec.get("device"),
            invert=bool(spec.get("invert", True)),  # True => reward = 1 - tox
        )
    raise ValueError(f"Unknown reward type: {t}")
```

That’s all for code changes.

---

# 2) Example JSON configs

**A) BLEU-only (clipped 0–1):**

```json
{
  "REWARD": {
    "type": "bleu",
    "smooth_method": "exp",
    "lowercase": true,
    "tokenize": "13a",
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

**B) ROUGE-L with EMA normalization:**

```json
{
  "REWARD": {
    "type": "rouge_l",
    "use_stemmer": true,
    "lowercase": true,
    "postprocess": [
      {"type": "ema_normalize", "decay": 0.995}
    ]
  }
}
```

**C) Low-toxicity shaping (reward = 1 − toxicity):**

```json
{
  "REWARD": {
    "type": "toxicity",
    "model": "original",      // or "unbiased"
    "invert": true,           // reward = 1 - tox
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

**D) Weighted mix: semantics + ROUGE-L + low-toxicity**

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "cosine_sts", "weight": 0.5},
      {"type": "rouge_l",    "weight": 0.4},
      {"type": "toxicity",   "invert": true, "weight": 0.1}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

---

# 3) Notes

* Install as needed:

  * `pip install sacrebleu`
  * `pip install rouge-score`
  * `pip install detoxify torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` (choose the right CUDA index for your env, or drop the index for CPU)
* **Toxicity** is local (no API). `Detoxify` models: `"original"`, `"unbiased"`, etc. We cache per device.
* All rewards return floats; post-processors (clip/scale/EMA normalize) compose cleanly with your **WeightedSum**.
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
