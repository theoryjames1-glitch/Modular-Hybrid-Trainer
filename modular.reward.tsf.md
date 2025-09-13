
# rewards.py (drop-in module)

```python
# rewards.py
import math, re, statistics, time
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# -----------------------------
# Base interface
# -----------------------------
class Reward:
    """Return a float reward in [unbounded], caller may post-process."""
    def __call__(self, response: str, target: str, *, prompt: str = "", item: dict | None = None) -> float:
        raise NotImplementedError

# -----------------------------
# Simple reward primitives
# -----------------------------
class ExactMatch(Reward):
    def __call__(self, response, target, **_):
        return 1.0 if response.strip() == target.strip() else 0.0

class KeywordCount(Reward):
    def __init__(self, keywords: List[str], normalize: bool = True):
        self.keywords = [k.lower() for k in keywords]
        self.normalize = normalize
    def __call__(self, response, target, **_):
        text = response.lower()
        hits = sum(text.count(k) for k in self.keywords)
        return hits / max(1, len(self.keywords)) if self.normalize else float(hits)

class RegexPresence(Reward):
    def __init__(self, pattern: str, flags: int = re.IGNORECASE):
        self.re = re.compile(pattern, flags)
    def __call__(self, response, target, **_):
        return 1.0 if self.re.search(response or "") else 0.0

class JaccardUnigram(Reward):
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
    def __call__(self, response, target, **_):
        a = response or ""
        b = target or ""
        if self.lowercase:
            a, b = a.lower(), b.lower()
        set_a, set_b = set(a.split()), set(b.split())
        if not set_a and not set_b: return 1.0
        if not set_a or not set_b:  return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / max(1, union)

class LengthWindow(Reward):
    """Reward 1.0 at center of window [min,max], decays linearly to 0 outside."""
    def __init__(self, min_tokens: int, max_tokens: int):
        self.min, self.max = int(min_tokens), int(max_tokens)
    def __call__(self, response, target, **_):
        n = len((response or "").split())
        if self.min <= n <= self.max: return 1.0
        # linear falloff: distance to nearest bound over window size
        if n < self.min:
            return max(0.0, 1.0 - (self.min - n) / max(1, self.max - self.min))
        return max(0.0, 1.0 - (n - self.max) / max(1, self.max - self.min))

# -----------------------------
# Semantic similarity (SentenceTransformers)
# -----------------------------
class CosineSTS(Reward):
    """Cosine similarity in [0,1] using SentenceTransformers; caches model & device."""
    _MODEL_CACHE: dict[str, SentenceTransformer] = {}
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        if not _HAS_ST:
            raise ImportError("SentenceTransformers not installed. pip install sentence-transformers")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_name not in CosineSTS._MODEL_CACHE:
            CosineSTS._MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=self.device)
        self.model = CosineSTS._MODEL_CACHE[model_name]
    @torch.inference_mode()
    def __call__(self, response, target, **_):
        e1 = self.model.encode(response or "", convert_to_tensor=True, normalize_embeddings=True)
        e2 = self.model.encode(target   or "", convert_to_tensor=True, normalize_embeddings=True)
        sim = st_util.cos_sim(e1, e2).item()  # [-1,1]
        return 0.5 * (sim + 1.0)              # map to [0,1]

# -----------------------------
# Post-processors / wrappers
# -----------------------------
class Clip(Reward):
    def __init__(self, base: Reward, lo: float = 0.0, hi: float = 1.0):
        self.base, self.lo, self.hi = base, float(lo), float(hi)
    def __call__(self, *args, **kwargs):
        return max(self.lo, min(self.hi, float(self.base(*args, **kwargs))))

class Scale(Reward):
    def __init__(self, base: Reward, mul: float = 1.0, add: float = 0.0):
        self.base, self.mul, self.add = base, float(mul), float(add)
    def __call__(self, *args, **kwargs):
        return self.mul * float(self.base(*args, **kwargs)) + self.add

class EMANormalize(Reward):
    """Keeps an exponential moving average of mean/std to normalize rewards online."""
    def __init__(self, base: Reward, decay: float = 0.99, eps: float = 1e-8):
        self.base, self.decay, self.eps = base, float(decay), float(eps)
        self._m, self._v = 0.0, 1.0  # mean, variance (EMAs)
        self._init = False
    def __call__(self, *args, **kwargs):
        x = float(self.base(*args, **kwargs))
        if not self._init:
            self._m, self._v, self._init = x, 1.0, True
        # EMA updates
        m = self.decay * self._m + (1 - self.decay) * x
        v = self.decay * self._v + (1 - self.decay) * (x - m) ** 2
        self._m, self._v = m, v
        return (x - m) / math.sqrt(v + self.eps)

# -----------------------------
# Combinators
# -----------------------------
class WeightedSum(Reward):
    def __init__(self, components: List[Tuple[Reward, float]]):
        assert components, "WeightedSum requires at least one component"
        self.components = components
        self.wsum = sum(w for _, w in components)
        self.wsum = self.wsum if self.wsum != 0 else 1.0
    def __call__(self, response, target, **kw):
        total = 0.0
        for comp, w in self.components:
            total += float(comp(response, target, **kw)) * float(w)
        return total / self.wsum

# -----------------------------
# Factory
# -----------------------------
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
    raise ValueError(f"Unknown reward type: {t}")

def _wrap_post(base: Reward, post: List[Dict[str, Any]] | None) -> Reward:
    if not post: return base
    r: Reward = base
    for p in post:
        pt = p.get("type", "").lower()
        if     pt == "clip":         r = Clip(r, lo=float(p.get("min", 0.0)), hi=float(p.get("max", 1.0)))
        elif   pt == "scale":        r = Scale(r, mul=float(p.get("mul", 1.0)), add=float(p.get("add", 0.0)))
        elif   pt == "ema_normalize":r = EMANormalize(r, decay=float(p.get("decay", 0.99)), eps=float(p.get("eps", 1e-8)))
        else:
            raise ValueError(f"Unknown postprocess type: {pt}")
    return r

def make_reward(spec: Dict[str, Any]) -> Reward:
    """
    spec examples:
      {"type":"cosine_sts","model":"all-MiniLM-L6-v2","postprocess":[{"type":"clip","min":0,"max":1}]}
      {"type":"weighted_sum","components":[
          {"type":"cosine_sts","weight":0.7},
          {"type":"regex_presence","pattern":"\\bthanks?\\b","weight":0.1},
          {"type":"length_window","min":40,"max":160,"weight":0.2}
      ], "postprocess":[{"type":"clip","min":0,"max":1}]}
    """
    t = spec.get("type", "").lower()
    if t == "weighted_sum":
        comps = []
        for c in spec.get("components", []):
            w = float(c.get("weight", 1.0))
            base = _build_leaf(c)
            base = _wrap_post(base, c.get("postprocess"))
            comps.append((base, w))
        reward = WeightedSum(comps)
        return _wrap_post(reward, spec.get("postprocess"))
    else:
        leaf = _build_leaf(spec)
        return _wrap_post(leaf, spec.get("postprocess"))
```

---

# Integrate in your PPO script

1. **Import and build** from JSON config:

```python
# train_ppo.py (your script)
from rewards import make_reward

# ...
cfg = load_config(cfg_path)
reward_spec = cfg.get("REWARD", {"type": "cosine_sts"})  # default
reward_fn = make_reward(reward_spec)
```

2. **Use it** inside the loop:

```python
# after you get response_text and target_text
rew = reward_fn(response_text, target_text, prompt=prompt, item=item)
batch_rewards.append(float(rew))
```

That’s it—now rewards are fully pluggable via your JSON.

---

# Example JSON configs

### A) Simple semantic similarity (clipped to \[0,1])

```json
{
  "REWARD": {
    "type": "cosine_sts",
    "model": "all-MiniLM-L6-v2",
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

### B) Weighted combo: semantics + regex + length shaping

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "cosine_sts", "weight": 0.7},
      {"type": "regex_presence", "pattern": "\\b(thanks|thank you)\\b", "weight": 0.1},
      {"type": "length_window", "min": 50, "max": 200, "weight": 0.2}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

### C) Keyword coverage with EMA normalization (for PPO stability)

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "keyword_count", "keywords": ["safety", "accuracy", "latency"], "normalize": true, "weight": 1.0}
    ],
    "postprocess": [
      {"type": "ema_normalize", "decay": 0.995}
    ]
  }
}
```

---

## Notes & tips

* All rewards are **pure-Python** and **GPU-optional** (only STS uses the GPU if available).
* You can nest post-processors both **inside** each component and **on the final mix**.
* If you want BLEU/ROUGE later, we can add optional leafs (sacrebleu/rouge-score) with the same factory pattern.
* For task-specific shaping, write a tiny `Reward` subclass and register it via `_build_leaf`.
