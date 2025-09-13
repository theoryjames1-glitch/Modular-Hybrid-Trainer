
## 🔹 Option A: Hugging Face sentiment pipeline

Simple and lightweight (uses a pretrained classifier like DistilBERT):

```python
# rewards.py (add near optional imports)
try:
    from transformers import pipeline
    _HAS_PIPELINE = True
except Exception:
    _HAS_PIPELINE = False


class SentimentReward(Reward):
    """
    Sentiment classifier reward.
    - model: HF model name (default 'distilbert-base-uncased-finetuned-sst-2-english')
    - target: 'positive' or 'negative' (which label to reward)
    - scale: if True, use probability directly; if False, return 1 if target label wins else 0.
    """
    _CACHE: dict[str, Any] = {}

    def __init__(self, model: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 target: str = "positive", scale: bool = True, device: int | str | None = None):
        if not _HAS_PIPELINE:
            raise ImportError("transformers required for SentimentReward (pipeline).")
        key = f"{model}|{device}"
        if key not in SentimentReward._CACHE:
            SentimentReward._CACHE[key] = pipeline("sentiment-analysis", model=model, device=0 if device=="cuda" else -1)
        self.analyzer = SentimentReward._CACHE[key]
        self.target = target.lower()
        self.scale = scale

    def __call__(self, response, target_text=None, **_):
        txt = (response or "").strip()
        if not txt:
            return 0.0
        result = self.analyzer(txt, truncation=True, max_length=512)[0]
        label = result["label"].lower()
        score = float(result["score"])
        if self.scale:
            return score if self.target in label else 1.0 - score
        else:
            return 1.0 if self.target in label else 0.0
```

---

## 🔹 Option B: Use `Detoxify` sentiment head

Detoxify only gives toxicity-related scores, so for *general positive/negative sentiment* you’ll want a classifier like the SST-2 model above.

---

## 🔹 Add to factory

In `_build_leaf`:

```python
    if t in ("sentiment", "sentiment_reward"):
        return SentimentReward(
            model=spec.get("model", "distilbert-base-uncased-finetuned-sst-2-english"),
            target=spec.get("target", "positive"),
            scale=bool(spec.get("scale", True)),
            device=spec.get("device"),
        )
```

---

## 🔹 Example JSON configs

**A) Reward positive sentiment:**

```json
{
  "REWARD": {
    "type": "sentiment",
    "target": "positive",
    "scale": true,
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

**B) Reward negative sentiment (binary, not scaled):**

```json
{
  "REWARD": {
    "type": "sentiment",
    "target": "negative",
    "scale": false
  }
}
```

**C) Blend sentiment + fluency (perplexity):**

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "sentiment", "target": "positive", "scale": true, "weight": 0.5},
      {"type": "perplexity", "model": "gpt2-medium", "weight": 0.5}
    ]
  }
}
```

---

✅ This way you can shape PPO to prefer **positive tone**, **negative tone**, or mix it with your other rewards (toxicity, BLEU, perplexity, etc).

