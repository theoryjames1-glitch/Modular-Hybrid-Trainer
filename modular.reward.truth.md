
# 🔩 Add to `rewards.py`

## Guarded imports (near the others)

```python
try:
    from transformers import pipeline as hf_pipeline
    _HAS_HF_PIPELINE = True
except Exception:
    _HAS_HF_PIPELINE = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util  # you already import this above
    _HAS_ST = True
except Exception:
    _HAS_ST = False
```

> You already have `SentenceTransformer` for STS—this keeps it consistent.

## NLI helper (softmax labels → probs)

```python
import math

def _softmax(xs):
    mx = max(xs)
    exps = [math.exp(x - mx) for x in xs]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]
```

## 1) Truthfulness via NLI against a reference

```python
class TruthfulnessNLI(Reward):
    """
    Estimate truthfulness of `response` by NLI against a provided `target` (reference string).
    Reward ~ P(entailment) - P(contradiction), mapped to [0,1].
    Models: 'roberta-large-mnli', 'microsoft/deberta-v3-large-mnli', etc.
    """
    _CACHE = {}

    def __init__(self, model: str = "roberta-large-mnli", device: str | None = None,
                 aggregation: str = "p_e_minus_p_c"):
        if not _HAS_HF_PIPELINE:
            raise ImportError("transformers pipeline required for TruthfulnessNLI")
        key = f"{model}|{device}"
        if key not in TruthfulnessNLI._CACHE:
            pipedev = 0 if (device == "cuda") else -1
            TruthfulnessNLI._CACHE[key] = hf_pipeline(
                "text-classification", model=model, return_all_scores=True, device=pipedev
            )
        self.clf = TruthfulnessNLI._CACHE[key]
        self.aggregation = aggregation

    def __call__(self, response: str, target: str, **_):
        hypothesis = (response or "").strip()
        premise    = (target   or "").strip()
        if not hypothesis or not premise:
            return 0.5  # neutral fallback

        # NLI expects premise/hypothesis pairs (some pipelines use "sequence" & "text")
        # Using "pair" form: pass as tuple (premise, hypothesis) if supported.
        out = self.clf({"text": premise, "text_pair": hypothesis})[0]  # list of dicts with 'label' and 'score'

        # Map to probs
        # Common labels: 'ENTAILMENT', 'NEUTRAL', 'CONTRADICTION' (case/name can vary by model)
        scores = {d["label"].lower(): float(d["score"]) for d in out}
        # If not already normalized, normalize (many pipelines already return probs).
        p = [scores.get("entailment", 0.0), scores.get("neutral", 0.0), scores.get("contradiction", 0.0)]
        s = sum(p) or 1.0
        p_e, p_n, p_c = p[0]/s, p[1]/s, p[2]/s

        if self.aggregation == "p_e_minus_p_c":
            # [-1,1] → [0,1]
            return max(0.0, min(1.0, 0.5 * (p_e - p_c) + 0.5))
        elif self.aggregation == "p_e":
            return p_e
        elif self.aggregation == "p_e_plus_half_neutral":
            return max(0.0, min(1.0, p_e + 0.5 * p_n))
        else:
            return p_e  # default
```

## 2) Retrieval-augmented truthfulness (local corpus → NLI)

```python
class TruthfulnessRAG(Reward):
    """
    Retrieval-augmented truthfulness: retrieve top-k evidence from a local corpus (list of strings),
    then aggregate NLI signals: reward ~ max_evidence P(entailment) - max_evidence P(contradiction).
    You must pass `corpus` at construction (list[str]) or later via set_corpus().
    """
    _NLI_CACHE = {}
    _EMB_CACHE = {}

    def __init__(self,
                 corpus: list[str] | None = None,
                 embedder: str = "all-MiniLM-L6-v2",
                 nli_model: str = "roberta-large-mnli",
                 device: str | None = None,
                 top_k: int = 5,
                 agg: str = "max"):
        if not _HAS_ST:
            raise ImportError("SentenceTransformers required for TruthfulnessRAG")
        if not _HAS_HF_PIPELINE:
            raise ImportError("transformers pipeline required for TruthfulnessRAG")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Embedder (cached per model)
        if embedder not in TruthfulnessRAG._EMB_CACHE:
            TruthfulnessRAG._EMB_CACHE[embedder] = SentenceTransformer(embedder, device=self.device)
        self.emb = TruthfulnessRAG._EMB_CACHE[embedder]

        # NLI pipeline (cached per model/device)
        nli_key = f"{nli_model}|{self.device}"
        if nli_key not in TruthfulnessRAG._NLI_CACHE:
            pipedev = 0 if self.device == "cuda" else -1
            TruthfulnessRAG._NLI_CACHE[nli_key] = hf_pipeline(
                "text-classification", model=nli_model, return_all_scores=True, device=pipedev
            )
        self.nli = TruthfulnessRAG._NLI_CACHE[nli_key]

        self.top_k = int(top_k)
        self.agg = agg  # 'max' or 'mean'
        self.set_corpus(corpus or [])

    def set_corpus(self, corpus: list[str]):
        self.corpus = [c for c in (corpus or []) if isinstance(c, str)]
        if self.corpus:
            self.corpus_emb = self.emb.encode(self.corpus, convert_to_tensor=True, normalize_embeddings=True)
        else:
            self.corpus_emb = None

    @torch.inference_mode()
    def __call__(self, response: str, target: str | None = None, **_):
        # If target (ground-truth) provided, we can append it to corpus temporarily as strong evidence.
        evidence_texts = []
        if self.corpus_emb is not None and len(self.corpus) > 0:
            q = self.emb.encode(response or "", convert_to_tensor=True, normalize_embeddings=True)
            sims = st_util.cos_sim(q, self.corpus_emb)[0]
            topk_idx = torch.topk(sims, k=min(self.top_k, len(self.corpus))).indices.tolist()
            evidence_texts = [self.corpus[i] for i in topk_idx]

        if target:
            evidence_texts = [target] + evidence_texts

        if not evidence_texts:
            return 0.5

        p_ent_list, p_con_list = [], []
        for ev in evidence_texts:
            out = self.nli({"text": ev, "text_pair": response})
            scores = {d["label"].lower(): float(d["score"]) for d in out[0]}
            p_e = scores.get("entailment", 0.0)
            p_c = scores.get("contradiction", 0.0)
            s = (p_e + p_c + scores.get("neutral", 0.0)) or 1.0
            p_ent_list.append(p_e / s)
            p_con_list.append(p_c / s)

        if self.agg == "mean":
            p_e, p_c = sum(p_ent_list)/len(p_ent_list), sum(p_con_list)/len(p_con_list)
        else:  # 'max'
            p_e, p_c = max(p_ent_list), max(p_con_list)

        # map [-1,1] → [0,1]
        return max(0.0, min(1.0, 0.5 * (p_e - p_c) + 0.5))
```

## 3) Wire into the factory

Add to `_build_leaf(spec)`:

```python
    if t in ("truth_nli", "truthfulness_nli", "veracity_nli"):
        return TruthfulnessNLI(
            model=spec.get("model", "roberta-large-mnli"),
            device=spec.get("device"),
            aggregation=spec.get("aggregation", "p_e_minus_p_c"),
        )
    if t in ("truth_rag", "truthfulness_rag", "veracity_rag"):
        # corpus can be provided right here (list of strings), or set later via your script
        return TruthfulnessRAG(
            corpus=spec.get("corpus", []),
            embedder=spec.get("embedder", "all-MiniLM-L6-v2"),
            nli_model=spec.get("nli_model", "roberta-large-mnli"),
            device=spec.get("device"),
            top_k=int(spec.get("top_k", 5)),
            agg=spec.get("agg", "max"),
        )
```

---

## 📦 Example JSON configs

**A) Truthfulness vs. known reference (per-sample `target`):**

```json
{
  "REWARD": {
    "type": "truth_nli",
    "model": "microsoft/deberta-v3-large-mnli",
    "aggregation": "p_e_minus_p_c",
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

**B) Retrieval-augmented truthfulness (local wiki-like corpus):**

```json
{
  "REWARD": {
    "type": "truth_rag",
    "embedder": "all-MiniLM-L6-v2",
    "nli_model": "roberta-large-mnli",
    "top_k": 5,
    "agg": "max",
    "corpus": [
      "Einstein developed the theory of relativity.",
      "The capital of France is Paris.",
      "Python is a high-level programming language."
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

> You can also omit `"corpus"` in JSON and call `reward_fn.set_corpus(your_list)` in your script after loading a local knowledge base.

**C) Blend truthfulness with toxicity & fluency (1/ppl):**

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "truth_nli", "model": "roberta-large-mnli", "weight": 0.6},
      {"type": "toxicity",  "invert": true, "weight": 0.1},
      {"type": "perplexity", "model": "gpt2-medium", "weight": 0.3}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

---

## ⚠️ Limitations (brief + honest)

* NLI ≠ perfect fact-checking. It judges **logical support** given your reference/evidence, so the quality of the evidence matters.
* For open-domain claims, RAG needs a **good corpus**. If the true fact isn’t in your corpus, the score may be misleading.
* Some NLI models are sensitive to phrasing; consider **paraphrase-robust evidence** and aggregation (top-k, max/mean).
