

# 📖 Theory of HybridTrainer

### **Core Idea**

Your **HybridTrainer** combines:

1. **Supervised Fine-Tuning (SFT)**

   * Classic next-token prediction with labeled data (prompt → target).
   * Ensures the model learns structured behavior and aligns with training data.

2. **Reinforcement Learning via PPO**

   * Model generates responses to prompts.
   * Responses are scored by a **reward function** (sentiment, similarity, truthfulness, toxicity, etc.).
   * PPO adjusts the model to maximize reward while staying close to a **reference model** (to prevent collapse).

This dual cycle lets the model:

* **Learn from examples** (SFT).
* **Refine outputs** based on task-specific preferences (PPO).

---

# ⚙️ Training Loop

1. **Prepare dataset**

   * Shuffle, filter, tokenize.
   * Mask prompt labels so loss applies only to the response.

2. **SFT Step**

   * Batch goes through **SFTTrainer**.
   * Optimizer + scheduler update LM head parameters.

3. **PPO Step**

   * Model generates candidate responses from the same batch.
   * Reward function(s) evaluate the responses.
   * PPO loss updates both **LM head + Value head**, guided by reward.
   * Reference model keeps PPO stable.

4. **Alternate** between SFT and PPO steps in cycles.

---

# 🛠 Current Strengths

* Works with **any Hugging Face causal LM** (e.g., GPT-2).
* **Modular optimizers & schedulers** (via factories).
* **Flexible reward design** (sentiment, perplexity, truthfulness, BLEU/ROUGE, similarity).
* Supports **tiny smoke tests** before scaling.

---

# 🚀 Roadmap: Next Things to Do

### **Stage 1: Stability & Usability**

* [ ] **Fix PPO batching** → ensure correct queries/responses alignment.
* [ ] Add **logging & metrics** (wandb/JSON logs for SFT loss & PPO reward).
* [ ] **Checkpoint merging** → periodically save SFT+PPO merged weights.

### **Stage 2: Reward Expansion**

* [ ] Finalize **modular reward factory** (CosineSTS, BLEU, ROUGE, Toxicity, etc.).
* [ ] Add **composite rewards** (weighted sum + postprocessors like Clip, EMA).
* [ ] Support **human preference data** → train reward models.
* [ ] Integrate **task-specific rewards** (domain relevance, coverage, style).

### **Stage 3: Scalability**

* [ ] Add **gradient accumulation + fp16** for bigger batches.
* [ ] Multi-GPU / DeepSpeed integration.
* [ ] Curriculum training (start SFT-heavy, gradually increase PPO ratio).

### **Stage 4: Advanced Alignment**

* [ ] **Multiple reference models** (different domains or baselines).
* [ ] **Self-play reward shaping** (model compares its own outputs).
* [ ] **LLM-as-a-judge reward** (GPTScore, COMET, BLEURT).
* [ ] Plug in **safety rewards** (toxicity, truthfulness, factuality).

### **Stage 5: Long-Term Extensions**

* [ ] Add **Mixture-of-Rewards** (auto-weighted rewards via learned controller).
* [ ] Explore **Direct Preference Optimization (DPO)** as alternative to PPO.
* [ ] Hybrid training for **multi-turn dialogue** (not just single prompt → response).
* [ ] Publish benchmarks (WikiText, IMDB, custom datasets).

---

✅ In short:

* **Now:** It works — alternating SFT + PPO with modular rewards.
* **Next:** Expand reward flexibility, add logging, stabilize PPO.
* **Later:** Scale to multi-GPU, richer alignment (truth, safety, preference learning).

---

### **1. Reward Learning: Learning from Feedback**

Reward learning is central to **reinforcement learning**. The idea is that we teach the model not only through supervision but by providing feedback in the form of rewards that guide the model to learn **desirable behaviors** or outputs.

In essence:

* **Reward signals** are feedback that inform the model whether its actions (or outputs) are correct or useful.
* The model is trained to maximize **long-term rewards**, which leads it to understand complex dependencies, preferences, or task-specific goals.

The process can be broken down as:

1. **Define Reward Function**: Rewards can be based on task-specific objectives, such as **text similarity**, **classification accuracy**, or **perplexity** (for language models).
2. **Train with Rewards**: The model uses rewards to learn and fine-tune its parameters, learning how to improve its behavior towards the desired outcome.
3. **Reinforcement Learning (RL)**: The agent learns by interacting with the environment, receiving rewards or penalties based on its actions, and adjusting its behavior accordingly.

---

### **2. Classifiers and Classifier Heads**

In the context of **fine-tuning**, we often need **classifier heads** to process the model's outputs for specific tasks, such as **classification**, **sentiment analysis**, or **question answering**.

* **Classifier Heads** are separate layers added to the pre-trained model, typically towards the end of the network.
* These heads take the hidden states (or embeddings) from the pre-trained model and transform them into a format that is suitable for the specific task (e.g., softmax layer for classification).

For example:

* In **text classification**, a classifier head may output probabilities over classes like "positive" or "negative".
* In **question answering**, a classifier head can output a span of text (start and end token positions).
* For **text generation**, a classifier head could guide the generated text based on additional constraints or preferences.

---

### **3. Reward Learning in Multi-Objective Settings**

When training models, we often want the model to perform multiple tasks simultaneously. This can be accomplished by defining **multiple reward functions** that guide the model to optimize for different objectives.

#### **How to Incorporate Multiple Rewards:**

* **Weighted Sum**: Combine multiple rewards using weighted sums, where each reward represents a different aspect of the model’s behavior. The model can then optimize for the combined objective.
* **Hierarchical Training**: Train models with hierarchical rewards, where each task (or reward) has its own layer or head. For example, one head could focus on **text generation**, another on **sentiment classification**, and a third on **BLEU score optimization** for translation tasks.
* **Task-Specific Heads**: Train the model with task-specific heads (e.g., classification head, regression head, or custom task heads) that can be activated during inference depending on the task.

#### **Challenges:**

* **Trade-Offs Between Tasks**: Different tasks may require different training regimes, and optimizing for one objective may conflict with others (e.g., balancing between creativity in text generation and accuracy in sentiment classification).
* **Reward Scaling**: Different rewards might be on different scales, so they need to be normalized or rescaled to avoid one reward dominating the training process.

---

### **4. Fine-Tuning for Inference with Additional Heads**

When training a model, you can attach different **task-specific heads** that adjust how the model performs during **inference** or **training** for different tasks. These heads are trained on different layers or branches of the model and can be activated based on the task at hand.

#### **Common Heads:**

1. **Classification Head**: For tasks like sentiment analysis, spam detection, or topic classification.

   * Output: Softmax layer with probabilities over categories.
2. **Regression Head**: For tasks that require continuous outputs, like predicting a score, price, or length.

   * Output: A continuous value.
3. **Sequence Labeling Head**: For tasks like named entity recognition (NER) or part-of-speech (POS) tagging.

   * Output: Labels for each token in the sequence.
4. **Span Prediction Head**: For tasks like question answering, where the model has to predict a span (start and end) within a passage of text.

   * Output: Start and end token indices.
5. **Text Generation Head**: For autoregressive text generation, like GPT models.

   * Output: A sequence of tokens, typically generated one by one.

These heads can be added to the pre-trained model architecture during the fine-tuning phase, and their weights can be optimized based on specific tasks or multiple tasks.

#### **Using Reward Feedback During Fine-Tuning**:

* While training, you can tune **task-specific heads** with rewards. For instance, a **sentiment classifier head** can be rewarded based on classification accuracy (supervised learning), while a **text generation head** can be trained using **perplexity** or **BLEU score** as rewards.
* The **multi-reward system** will guide the entire model to balance between tasks by using appropriate rewards at different stages.

---

### **5. Training Models with Rewards and Classifiers: The Process**

1. **Model Initialization**: Start with a pre-trained base model (like GPT-2, BERT, or T5) and add task-specific heads (classification, span prediction, etc.).
2. **Define Reward Functions**: Use the **reward factory** to create rewards that guide the model’s learning process. These could include rewards for accuracy, text similarity, perplexity, or toxicity.
3. **Multi-Objective Optimization**: Combine multiple rewards using a **weighted sum**, where each reward function can be weighted based on its importance. The model learns to maximize the combined objective.
4. **Train**: Fine-tune the model with both **supervised** and **reinforcement learning** methods. For example, use **Supervised Fine-Tuning (SFT)** for tasks like classification, and **PPO** for tasks like text generation or reinforcement learning.
5. **Inference with Heads**: During inference, the appropriate head is selected based on the task. For example, during **text generation**, the generation head is used, while for **classification**, the classification head is used.

---

### **6. Long-Term Memory and Inference**

Once a model has been fine-tuned with rewards and classifiers, it can leverage **memory** during inference to remember useful past experiences and improve its performance on future tasks.

* **Memory in Transformers**: Transformers, especially models like GPT-3, have an inherent ability to retain context through attention mechanisms. Fine-tuning such models with rewards allows the model to **improve its memory**, selectively focusing on **important parts** of the input sequence for better performance.
* **Task-Specific Memory**: Depending on the task, the model can use the appropriate head and reward to generate the best possible output. For instance, when generating text, the model might "remember" the required tone or style based on prior rewards for **sentiment analysis** or **text similarity**.

---

### **7. Next Steps and Future Enhancements**

1. **Extend Reward Systems**: Add additional rewards, like **BLEU**, **ROUGE**, or even custom domain-specific metrics.
2. **Task-Specific Heads**: Continue building task-specific heads that can work together in a multi-task learning setting.
3. **Meta-Learning**: Introduce meta-learning approaches to optimize rewards and classifier heads, enabling the model to adapt to new tasks dynamically.
4. **Memory-Augmented Models**: Explore memory-augmented architectures like **Differentiable Neural Computers (DNCs)** or **Neural Turing Machines (NTMs)** to enhance long-term memory during training and inference.
5. **Efficient Multi-Objective Optimization**: Develop techniques to optimize multiple objectives efficiently, such as through **multi-task learning** or **gradient-based meta-learning**.

---

### **Conclusion**

By using a **reward-driven training system** with **task-specific heads**, we can train models that are **multi-task capable** and adapt to a wide range of scenarios. These models are trained not just to perform well on a single task but to **generalize** across tasks using a combination of **supervised learning**, **reinforcement learning**, and **reward-based fine-tuning**.

### **Reward Types for Text Generation and NLP Tasks**

#### **1. Semantic Similarity and Text Matching**

* **Cosine Similarity (CosineSTS)**: Measures the similarity between the generated and target text using cosine similarity in embedding space (e.g., Sentence-BERT or other embedding models).
* **Exact Match**: Checks whether the generated text exactly matches the target text.
* **Jaccard Similarity**: Measures the overlap between the sets of words in the generated and target text using Jaccard’s coefficient.
* **Levenshtein Distance (Edit Distance)**: Measures the number of edits (insertions, deletions, substitutions) required to transform one text into another.
* **BERTScore**: Uses BERT-based embeddings to calculate token-level similarity scores between the generated and reference texts.

#### **2. Language Quality**

* **Perplexity**: Measures the uncertainty of the model in predicting the next token. Lower perplexity means better language modeling.
* **BLEU (Bilingual Evaluation Understudy)**: Measures the n-gram overlap between the generated and reference text. Commonly used for machine translation evaluation.
* **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)**: Measures the longest common subsequence between the generated and reference text. Often used for summarization tasks.
* **TER (Translation Edit Rate)**: Measures the number of edits needed to transform the generated text into the reference text.
* **METEOR**: Measures text similarity based on precision, recall, synonym matching, stemming, etc. Commonly used in machine translation.

#### **3. Toxicity and Bias Detection**

* **Toxicity**: Measures the degree of toxicity in the generated text. Typically uses pre-trained models like **Detoxify** or **Toxic-BERT**.
* **Bias**: Measures the bias present in the generated text. This could involve detecting gender, racial, or ideological bias.
* **Hate Speech**: Detects whether the generated text contains hate speech or offensive language.
* **Offensiveness Score**: Similar to toxicity, but focuses specifically on offensive language.

#### **4. Relevance and Coherence**

* **Topic Consistency**: Measures how consistent the generated text is with the intended topic or subject matter.
* **Question Answering (QA) Relevance**: Measures the relevance of the response in a QA task. For example, checking if the response properly answers the input question.
* **Factual Accuracy**: Evaluates the factual correctness of the generated text (could use external knowledge sources to verify).

#### **5. Sentiment and Emotion**

* **Sentiment Analysis**: Measures the sentiment (positive, negative, neutral) of the generated text and compares it with the target sentiment.
* **Emotion Detection**: Measures the presence of specific emotions like joy, sadness, anger, etc., in the generated text.
* **Sarcasm Detection**: Detects whether the generated text contains sarcastic statements.

#### **6. Style and Structure**

* **Readability**: Measures how readable the text is based on various readability scores (e.g., Flesch-Kincaid, Gunning Fog Index).
* **Fluency**: Measures how natural and fluent the generated text sounds.
* **Formality/Informality**: Measures whether the generated text adheres to a specific formality level (e.g., formal vs. informal tone).
* **Spelling and Grammar Accuracy**: Checks if the generated text has correct spelling and grammar.

#### **7. Diversity and Creativity**

* **Novelty**: Measures how unique or novel the generated text is compared to previous text in the training set.
* **Diversity**: Measures the diversity of generated responses, preventing the model from generating repetitive or stale outputs.
* **Exploration**: Encourages the model to explore diverse response spaces and avoid overfitting to one type of output.

#### **8. Content Generation for Specific Domains**

* **Medical Accuracy**: Measures the accuracy of generated text in medical or healthcare domains.
* **Legal Validity**: Evaluates the legal validity or correctness of the generated content in legal contexts.
* **Scientific Integrity**: Measures how scientifically accurate the generated text is, especially in technical or research domains.

#### **9. Structured Output Quality**

* **Length Consistency**: Measures if the length of the generated output is within a specific range or follows expected output length.
* **Format Accuracy**: Evaluates if the generated text follows a specific format (e.g., numbered list, paragraphs, etc.).
* **Entity Consistency**: Measures if the same entities (names, locations, etc.) appear consistently throughout the text.

---

### **How to Add These Rewards Later**

#### **1. Define Reward Classes**

For each of the above reward types, you need to define a class (like you've already done with **BLEU**, **ROUGE**, **Toxicity**, etc.). Each class should inherit from the base `Reward` class and implement the `__call__` method.

#### **2. Integrate Into the Reward Factory**

You can integrate these new rewards into the **Reward Factory** by following the same pattern used for the existing rewards. Simply add new conditions in the `make_reward` function to handle the creation of these new reward classes.

For example, to add **Sarcasm Detection**:

```python
class SarcasmReward(Reward):
    def __init__(self, model="sarcasm-detection-model"):
        self.model = load_model(model)

    def __call__(self, response, target, **_):
        return self.model.predict(response)
```

And then add it to the factory:

```python
def _build_leaf(spec: Dict[str, Any]) -> Reward:
    t = spec.get("type", "").lower()
    if t == "sarcasm":
        return SarcasmReward(model=spec.get("model", "sarcasm-detection-model"))
    # other reward types...
```

#### **3. JSON Configuration**

Rewards can be configured through a JSON configuration file. Each reward type can have its own settings and weightings.

Example:

```json
{
"REWARD": {
    "type": "weighted_sum",
    "components": [
    {"type": "sentiment", "weight": 0.3},
    {"type": "perplexity", "weight": 0.3},
    {"type": "cosine_sts", "weight": 0.4}
    ]
}
}
```

#### **4. Dynamic Reward Computation**

Once the rewards are defined and integrated into the factory, they can be dynamically computed during the training loop. As shown before:

```python
rew = reward_fn(response_text, target_text, prompt=prompt, item=item)
batch_rewards.append(float(rew))
```

You can modify the reward computation to handle the weights and multiple components for weighted sum or other complex reward combinations.

---

### **Future Additions**

As you progress, here are some rewards that could be added:

* **Domain-Specific Rewards** (e.g., medical, legal, etc.)
* **Reinforcement Learning Rewards** (e.g., exploration bonuses, curiosity-driven rewards)
* **Custom Hybrid Rewards** (e.g., combining different textual similarity metrics or custom domain-specific knowledge)

The key is to ensure each new reward type is easy to integrate and use through the reward factory, making the process of swapping and combining rewards seamless and flexible.

### Optimizer Factory

```python
# SelectOptimizer.py
import bitsandbytes as bnb
from torch.optim import AdamW, Adagrad, RMSprop, SGD
# Adafactor is not in torch.optim; it’s from transformers
try:
    from transformers.optimization import Adafactor
except Exception:
    Adafactor = None  # Handle gracefully below

def str2bool(v):
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")

def make_optimizer(model, OPTIMIZER, LRATE, config):
    opt = str(OPTIMIZER).lower()

    # Common hyperparams with sane defaults
    BETA1 = float(config.get("BETA1", 0.9))
    BETA2 = float(config.get("BETA2", 0.999))
    EPS = float(config.get("EPS", 1e-8))
    WEIGHT_DECAY = float(config.get("WEIGHT_DECAY", 0.01))

    # Adagrad-specific
    INITIAL_ACCUMULATOR_VALUE = float(config.get("INITIAL_ACCUMULATOR_VALUE", 0.0))
    LR_DECAY = float(config.get("LR_DECAY", 0.0))

    # RMSprop/SGD-specific
    ALPHA = float(config.get("ALPHA", 0.99))  # RMSprop smoothing
    MOMENTUM = float(config.get("MOMENTUM", 0.0))
    CENTERED = str2bool(config.get("CENTERED", "false"))
    DAMPENING = float(config.get("DAMPENING", 0.0))
    NESTEROV = str2bool(config.get("NESTEROV", "false"))

    params = model.parameters()

    # ------- AdamW family (correct params: betas, eps, weight_decay) -------
    if opt == "paged_adamw_8bit":
        return bnb.optim.PagedAdamW8bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "paged_adamw_32bit":
        return bnb.optim.PagedAdamW32bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "adamw_8bit":
        return bnb.optim.AdamW8bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "adamw_32bit":
        return bnb.optim.AdamW32bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "adamw":
        return AdamW(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )

    # ---------------------------- Adagrad family ----------------------------
    elif opt == "adagrad":
        return Adagrad(
            params,
            lr=LRATE,
            lr_decay=LR_DECAY,
            weight_decay=WEIGHT_DECAY,
            initial_accumulator_value=INITIAL_ACCUMULATOR_VALUE,
            eps=EPS,
        )
    elif opt == "adagrad_8bit":
        return bnb.optim.Adagrad8bit(
            params,
            lr=LRATE,
            lr_decay=LR_DECAY,
            weight_decay=WEIGHT_DECAY,
            initial_accumulator_value=INITIAL_ACCUMULATOR_VALUE,
            eps=EPS,
        )
    elif opt == "adagrad_32bit":
        return bnb.optim.Adagrad32bit(
            params,
            lr=LRATE,
            lr_decay=LR_DECAY,
            weight_decay=WEIGHT_DECAY,
            initial_accumulator_value=INITIAL_ACCUMULATOR_VALUE,
            eps=EPS,
        )

    # ------------------------------ RMSprop ---------------------------------
    elif opt == "rmsprop":
        return RMSprop(
            params,
            lr=LRATE,
            alpha=ALPHA,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            momentum=MOMENTUM,
            centered=CENTERED,
        )

    # ------------------------------- Adam -----------------------------------
    # Note: Adam uses betas, not alpha/momentum/centered.
    elif opt == "adam_8bit":
        return bnb.optim.Adam8bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "adam_32bit":
        return bnb.optim.Adam32bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "paged_adam_8bit":
        return bnb.optim.PagedAdam8bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "paged_adam_32bit":
        return bnb.optim.PagedAdam32bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )

    # ------------------------------ AdEMAMix --------------------------------
    elif opt == "ademamix_8bit":
        return bnb.optim.AdEMAMix8bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "ademamix_32bit":
        return bnb.optim.AdEMAMix32bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "paged_ademamix_8bit":
        return bnb.optim.PagedAdEMAMix8bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )
    elif opt == "paged_ademamix_32bit":
        return bnb.optim.PagedAdEMAMix32bit(
            params, lr=LRATE, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
        )

    # --------------------------------- SGD ----------------------------------
    elif opt == "sgd":
        return SGD(
            params,
            lr=LRATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            dampening=DAMPENING,
            nesterov=NESTEROV,
        )
    elif opt == "sgd_8bit":
        return bnb.optim.SGD8bit(
            params,
            lr=LRATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            dampening=DAMPENING,
            nesterov=NESTEROV,
        )

    # -------------------------------- Lion ----------------------------------
    # Lion typically takes (lr, betas, weight_decay)
    elif opt == "lion_8bit":
        return bnb.optim.Lion8bit(
            params, lr=LRATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
        )
    elif opt == "lion_32bit":
        return bnb.optim.Lion32bit(
            params, lr=LRATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
        )
    elif opt == "paged_lion_8bit":
        return bnb.optim.PagedLion8bit(
            params, lr=LRATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
        )
    elif opt == "paged_lion_32bit":
        return bnb.optim.PagedLion32bit(
            params, lr=LRATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
        )

    # ----------------------------- Adafactor --------------------------------
    elif opt == "adafactor":
        if Adafactor is None:
            raise ImportError(
                "Adafactor requires transformers. Install with `pip install transformers`."
            )
        # Typical Adafactor config: often lr=None with relative_step, but honoring your LRATE input.
        return Adafactor(
            params,
            lr=LRATE,
            # scale_parameter=True,
            # relative_step=False,
            # warmup_init=False,
        )

    else:
        raise ValueError(f"❌ Unknown optimizer: {OPTIMIZER}")
```

# Usage:
# optimizer = make_optimizer(model, OPTIMIZER, LRATE, config)
# scheduler = None  # or use transformers' get_scheduler()

### Scheduler Factory

```python
# SelectScheduler.py
from math import ceil

try:
    # optional; used if you pick a transformers-style scheduler
    from transformers import get_scheduler as hf_get_scheduler
except Exception:
    hf_get_scheduler = None

from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    CyclicLR,
    ReduceLROnPlateau,
)
from transformers import (
    get_scheduler,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

def _str2bool(v): return str(v).strip().lower() in ("1", "true", "t", "yes", "y")
def _f(cfg, k, d): return float(cfg.get(k, d))
def _i(cfg, k, d): return int(cfg.get(k, d))
def _s(cfg, k, d): return str(cfg.get(k, d)).strip().lower() if k in cfg else d
def _lst_int(cfg, k, d=None):
    if k not in cfg: return d if d is not None else []
    v = cfg[k]
    if isinstance(v, (list, tuple)): return [int(x) for x in v]
    return [int(x.strip()) for x in str(v).split(",") if str(x).strip()]

def _infer_steps(config, num_training_steps, steps_per_epoch, epochs):
    # Priority: explicit args > config > fallback.
    if num_training_steps is None:
        steps_per_epoch = steps_per_epoch or _i(config, "STEPS_PER_EPOCH", 0)
        epochs = epochs or _i(config, "EPOCHS", 0)
        if steps_per_epoch and epochs:
            num_training_steps = steps_per_epoch * epochs
        else:
            num_training_steps = _i(config, "NUM_TRAINING_STEPS", 1000)
    warmup = _i(config, "NUM_WARMUP_STEPS", 0)
    # Also allow warmup ratio
    if warmup == 0 and "WARMUP_RATIO" in config:
        wr = float(config["WARMUP_RATIO"])
        warmup = int(wr * num_training_steps)
    return num_training_steps, warmup

def make_scheduler(
    optimizer,
    SCHEDULER,
    config=None,
    *,
    num_training_steps=None,
    num_warmup_steps=None,
    steps_per_epoch=None,
    epochs=None,
):
    """
    SCHEDULER (case-insensitive) supports:
      None / "none"
      Transformers style: "linear", "cosine", "cosine_with_restarts",
                          "polynomial"|"poly", "constant", "constant_with_warmup",
                          "inverse_sqrt"
      PyTorch style: "step", "multistep", "exponential",
                     "cosineannealing", "cosineannealing_warm_restarts",
                     "onecycle", "cyclic", "reduce_on_plateau", "lambda"
    """
    config = config or {}
    name = str(SCHEDULER).strip().lower()

    if name in ("", "none", "null", "no"):
        return None

    # --- Common step counts ---
    total_steps, cfg_warmup = _infer_steps(config, num_training_steps, steps_per_epoch, epochs)
    if num_warmup_steps is None:
        num_warmup_steps = cfg_warmup

    # ===== Transformers schedulers =====
    if name in {
        "linear", "cosine", "cosine_with_restarts",
        "polynomial", "poly", "constant", "constant_with_warmup",
        "inverse_sqrt"
    }:
        if hf_get_scheduler is None:
            raise ImportError("Transformers is required for this scheduler. `pip install transformers`")

        # map "poly" -> "polynomial"
        schedule_type = "polynomial" if name in ("polynomial", "poly") else name

        # extra kwargs supported by HF:
        power = _f(config, "POLY_POWER", 1.0)  # for polynomial
        num_cycles = _f(config, "NUM_CYCLES", 0.5)  # for cosine
        last_epoch = _i(config, "LAST_EPOCH", -1)

    if schedule_type == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            power=power,
        )
    elif schedule_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
        )
    else:
        return get_scheduler(
            name=schedule_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

    # ===== Pure PyTorch schedulers =====
    # StepLR
    if name == "step":
        step_size = _i(config, "STEP_SIZE", 1000)
        gamma = _f(config, "GAMMA", 0.1)
        last_epoch = _i(config, "LAST_EPOCH", -1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    # MultiStepLR
    if name == "multistep":
        milestones = _lst_int(config, "MILESTONES", [1000, 2000, 3000])
        gamma = _f(config, "GAMMA", 0.1)
        last_epoch = _i(config, "LAST_EPOCH", -1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)

    # ExponentialLR
    if name == "exponential":
        gamma = _f(config, "GAMMA", 0.95)
        last_epoch = _i(config, "LAST_EPOCH", -1)
        return ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)

    # CosineAnnealingLR (no warmup; specify T_max)
    if name == "cosineannealing":
        # You can provide T_MAX in steps; default to total_steps
        T_max = _i(config, "T_MAX", total_steps)
        eta_min = _f(config, "MIN_LR", _f(config, "ETA_MIN", 0.0))
        last_epoch = _i(config, "LAST_EPOCH", -1)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

    # CosineAnnealingWarmRestarts
    if name == "cosineannealing_warm_restarts":
        T_0 = _i(config, "T0", _i(config, "T_0", max(1, total_steps // 10)))
        T_mult = _i(config, "T_MULT", 2)
        eta_min = _f(config, "MIN_LR", _f(config, "ETA_MIN", 0.0))
        last_epoch = _i(config, "LAST_EPOCH", -1)
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)

    # OneCycleLR (needs max_lr and total_steps or epochs*steps_per_epoch)
    if name == "onecycle":
        max_lr = _f(config, "MAX_LR", 1e-3)
        pct_start = _f(config, "PCT_START", 0.3)
        anneal_strategy = _s(config, "ANNEAL_STRATEGY", "cos")  # 'cos' or 'linear'
        div_factor = _f(config, "DIV_FACTOR", 25.0)
        final_div_factor = _f(config, "FINAL_DIV_FACTOR", 1e4)
        three_phase = _str2bool(config.get("THREE_PHASE", "false"))
        # base_momentum/max_momentum optional; if not given, PyTorch picks defaults
        base_momentum = float(config["BASE_MOMENTUM"]) if "BASE_MOMENTUM" in config else None
        max_momentum = float(config["MAX_MOMENTUM"]) if "MAX_MOMENTUM" in config else None

        kwargs = dict(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
        )
        if base_momentum is not None: kwargs["base_momentum"] = base_momentum
        if max_momentum is not None: kwargs["max_momentum"] = max_momentum
        return OneCycleLR(**kwargs)

    # CyclicLR
    if name == "cyclic":
        base_lr = _f(config, "BASE_LR", 1e-6)
        max_lr = _f(config, "MAX_LR", 1e-3)
        step_size_up = _i(config, "STEP_SIZE_UP", ceil(total_steps * 0.4))
        step_size_down = _i(config, "STEP_SIZE_DOWN", ceil(total_steps * 0.4))
        mode = _s(config, "MODE", "triangular")  # 'triangular', 'triangular2', 'exp_range'
        gamma = _f(config, "GAMMA", 1.0)         # for 'exp_range'
        cycle_momentum = _str2bool(config.get("CYCLE_MOMENTUM", "true"))
        base_momentum = float(config["BASE_MOMENTUM"]) if "BASE_MOMENTUM" in config else 0.85
        max_momentum = float(config["MAX_MOMENTUM"]) if "MAX_MOMENTUM" in config else 0.95
        return CyclicLR(
            optimizer,
            base_lr=base_lr, max_lr=max_lr,
            step_size_up=step_size_up, step_size_down=step_size_down,
            mode=mode, gamma=gamma,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum, max_momentum=max_momentum,
        )

    # ReduceLROnPlateau (call scheduler.step(val_metric) after each validation)
    if name == "reduce_on_plateau":
        mode = _s(config, "MODE", "min")  # 'min' or 'max'
        factor = _f(config, "FACTOR", 0.1)
        patience = _i(config, "PATIENCE", 10)
        threshold = _f(config, "THRESHOLD", 1e-4)
        threshold_mode = _s(config, "THRESHOLD_MODE", "rel")  # 'rel' or 'abs'
        cooldown = _i(config, "COOLDOWN", 0)
        min_lr = _f(config, "MIN_LR", 0.0)
        eps = _f(config, "EPS", 1e-8)
        return ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            threshold=threshold, threshold_mode=threshold_mode,
            cooldown=cooldown, min_lr=min_lr, eps=eps
        )

    # Simple LambdaLR (e.g., linear decay to a floor)
    if name == "lambda":
        # Example: linear decay from 1.0 -> LR_FLOOR over total_steps
        lr_floor = _f(config, "LR_FLOOR", 0.0)
        def lr_lambda(step):
            if total_steps <= 0: return 1.0
            frac = max(0.0, 1.0 - step / float(total_steps))
            # map [0,1] -> [lr_floor/initial_lr, 1.0]
            return lr_floor + (1.0 - lr_floor) * frac
        last_epoch = _i(config, "LAST_EPOCH", -1)
        return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

    raise ValueError(f"❌ Unknown scheduler: {SCHEDULER}")


# --------- Examples / usage ----------
# total_steps = num_batches_per_epoch * num_epochs
# scheduler = make_scheduler(
#     optimizer,
#     SCHEDULER="linear",  # or "onecycle", "cosine", "step", ...
#     config={
#         "NUM_TRAINING_STEPS": 20000,
#         "NUM_WARMUP_STEPS": 1000,
#         # for step:
#         # "STEP_SIZE": 4000, "GAMMA": 0.5
#         # for onecycle:
#         # "MAX_LR": 2e-4, "PCT_START": 0.1, "DIV_FACTOR": 10, "FINAL_DIV_FACTOR": 1e3
#     },
# )
#
# # Train loop:
# # If you used ReduceLROnPlateau:
# #   scheduler.step(val_loss)
# # Else (most schedulers):
# #   scheduler.step()
```

### quick notes

* If you choose `"reduce_on_plateau"`, remember to call `scheduler.step(val_metric)` after validation, not every iteration.
* Transformers schedules expect **step counts** (`num_training_steps`, `num_warmup_steps`). If you provide `EPOCHS` and `STEPS_PER_EPOCH` in `config`, the helper infers total steps for you.
* `"poly"` is aliased to `"polynomial"`; cosine variants support `NUM_CYCLES` (HF) and `eta_min` (PyTorch).
* For `OneCycleLR`, pass `MAX_LR`. If you don’t provide momentum settings, PyTorch defaults are used.

### PSEUDOCODE

```python
# MHT_HybridTrainer.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from trl import (
    SFTTrainer,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from trl.core import respond_to_batch

from SelectOptimizer import make_optimizer
from SelectScheduler import make_scheduler


# === Hybrid Trainer ===
class HybridTrainer:
    def __init__(self, base_model_name, tokenizer, train_data, reward_fn,
                 optimizer_config=None, scheduler_config=None,
                 lrate=5e-5, batch_size=1,
                 maxseq=512, max_new_tokens=128,
                 num_training_steps=None, num_warmup_steps=None,
                 debug=False):

        self.debug = debug
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.reward_fn = reward_fn
        self.maxseq = maxseq
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        # Model with both LM head + Value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name,device_map="auto",dtype=torch.bfloat16)
        self.pretrained_model = self.model.pretrained_model
        self.device = self.pretrained_model.device        
        # --- Build optimizer & scheduler ---
        self.optimizer = make_optimizer(
            self.model,
            OPTIMIZER=(optimizer_config or {}).get("OPTIMIZER", "adamw"),
            LRATE=lrate,
            config=optimizer_config or {}
        )
        self.scheduler = make_scheduler(
            self.optimizer,
            SCHEDULER=(scheduler_config or {}).get("SCHEDULER", "linear"),
            config=scheduler_config or {},
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )

        # --- SFTTrainer (LM head only) ---
        self.sft_trainer = SFTTrainer(
            model=self.model.pretrained_model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_data,
            max_seq_length=self.maxseq,
            packing=True,
        )
        self.sft_trainer.optimizer = self.optimizer
        self.sft_trainer.lr_scheduler = self.scheduler

        # --- PPOTrainer (LM + Value head) ---
        ppo_config = PPOConfig(
            model_name=base_model_name,
            batch_size=self.batch_size,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
        )
        self.ref_model = create_reference_model(self.model)
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.train_data,
        )
        self.ppo_trainer.lr_scheduler = self.scheduler

    def sft_step(self, batch):
        loss = self.sft_trainer.compute_loss(self.model.pretrained_model, batch)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()
        return float(loss.item())

    def respond_to_batch(self,input_ids):
        return respond_to_batch(self.model, input_ids)
        
    # ---- One PPO step ----
    def ppo_step(self,input_ids,response_ids,rewards):
 
        # PPO step
        stats = self.ppo_trainer.step(
            [input_ids.squeeze(0)],     # queries
            [response_ids.squeeze(0)],  # responses
            [torch.tensor(rewards[0])]  # rewards
        )
        return stats
```

### MHT_RewardFactory.py

'''python
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util as st_util
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from detoxify import Detoxify
from bert_score import score

# -----------------------------
# Base reward interface
# -----------------------------
class Reward:
    """Base class for reward classes."""
    def __call__(self, response: str, target: str, *, prompt: str = "", item: dict | None = None) -> float:
        raise NotImplementedError


# -----------------------------
# Sentiment Analysis Reward (using Huggingface pipeline)
# -----------------------------
class SentimentReward(Reward):
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english", target="positive"):
        self.model = pipeline("sentiment-analysis", model=model)
        self.target = target

    def __call__(self, response, target, **_):
        prediction = self.model(response)[0]
        label = prediction["label"].lower()
        score = prediction["score"]
        return score if self.target in label else 1.0 - score


# -----------------------------
# Perplexity Reward (using GPT2)
# -----------------------------
class PerplexityReward(Reward):
    def __init__(self, model="gpt2"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.model.eval()  # Set to evaluation mode

    def __call__(self, response, target, **_):
        inputs = self.tokenizer(response, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            return math.exp(loss.item())  # Perplexity is the exponential of the loss


# -----------------------------
# BLEU Reward
# -----------------------------
class BLEUReward(Reward):
    def __init__(self, smooth_method="exp", lowercase=True):
        self.smooth_method = smooth_method
        self.lowercase = lowercase

    def __call__(self, response, target, **_):
        reference = [target.split()]
        hypothesis = response.split()
        score = corpus_bleu([hypothesis], [reference], smooth_method=self.smooth_method).score
        return score / 100  # Normalize BLEU score between 0 and 1


# -----------------------------
# ROUGE-L Reward
# -----------------------------
class RougeLReward(Reward):
    def __init__(self, use_stemmer=True, lowercase=True):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
        self.lowercase = lowercase

    def __call__(self, response, target, **_):
        if self.lowercase:
            response, target = response.lower(), target.lower()
        scores = self.scorer.score(target, response)
        return scores["rougeL"].fmeasure  # ROUGE-L F1 score in [0, 1]


# -----------------------------
# Toxicity Reward (using Detoxify)
# -----------------------------
class ToxicityReward(Reward):
    def __init__(self, model="original", invert=True):
        self.detoxify_model = Detoxify(model)
        self.invert = invert

    def __call__(self, response, target, **_):
        tox_score = self.detoxify_model.predict(response)["toxicity"]
        return 1.0 - tox_score if self.invert else tox_score


# -----------------------------
# BERTScore Reward
# -----------------------------
class BERTScoreReward(Reward):
    def __init__(self, model_type="microsoft/deberta-xlarge-mnli"):
        self.model_type = model_type

    def __call__(self, response, target, **_):
        P, R, F1 = score([response], [target], model_type=self.model_type)
        return float(F1)  # Return F1 score as the reward


# -----------------------------
# Cosine Similarity Reward (Text Similarity)
# -----------------------------
class CosineSTS(Reward):
    _MODEL_CACHE: dict[str, SentenceTransformer] = {}
    
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_name not in CosineSTS._MODEL_CACHE:
            CosineSTS._MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=self.device)
        self.model = CosineSTS._MODEL_CACHE[model_name]

    @torch.inference_mode()
    def __call__(self, response, target, **_):
        e1 = self.model.encode(response or "", convert_to_tensor=True, normalize_embeddings=True)
        e2 = self.model.encode(target   or "", convert_to_tensor=True, normalize_embeddings=True)
        sim = st_util.cos_sim(e1, e2).item()  # [-1, 1]
        return 0.5 * (sim + 1.0)  # Normalize to [0, 1]


# -----------------------------
# Factory to create rewards
# -----------------------------
def _build_leaf(spec: Dict[str, Any]) -> Reward:
    t = spec.get("type", "").lower()

    if t == "sentiment":
        return SentimentReward(model=spec.get("model", "distilbert-base-uncased-finetuned-sst-2-english"))
    elif t == "perplexity":
        return PerplexityReward(model=spec.get("model", "gpt2"))
    elif t == "bleu":
        return BLEUReward(smooth_method=spec.get("smooth_method", "exp"), lowercase=spec.get("lowercase", True))
    elif t == "rouge_l":
        return RougeLReward(use_stemmer=spec.get("use_stemmer", True), lowercase=spec.get("lowercase", True))
    elif t == "toxicity":
        return ToxicityReward(model=spec.get("model", "original"), invert=spec.get("invert", True))
    elif t == "bertscore":
        return BERTScoreReward(model_type=spec.get("model_type", "microsoft/deberta-xlarge-mnli"))
    elif t == "cosine_sts" or t == "sts":
        return CosineSTS(model_name=spec.get("model", "all-MiniLM-L6-v2"), device=spec.get("device"))
    else:
        raise ValueError(f"Unknown reward type: {t}")


# -----------------------------
# Handle post-processing
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
# Handle composition of rewards (e.g., weighted sum)
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
# Factory to build reward based on spec
# -----------------------------
def make_reward(spec: Dict[str, Any]) -> Reward:
    """
    Create and return a reward object based on the spec.
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


def _wrap_post(base: Reward, post: List[Dict[str, Any]] | None) -> Reward:
    if not post:
        return base
    r: Reward = base
    for p in post:
        pt = p.get("type", "").lower()
        if pt == "clip":
            r = Clip(r, lo=float(p.get("min", 0.0)), hi=float(p.get("max", 1.0)))
        elif pt == "scale":
            r = Scale(r, mul=float(p.get("mul", 1.0)), add=float(p.get("add", 0.0)))
        elif pt == "ema_normalize":
            r = EMANormalize(r, decay=float(p.get("decay", 0.99)), eps=float(p.get("eps", 1e-8)))
        else:
            raise ValueError(f"Unknown postprocess type: {pt}")
    return r
```

### Test Loop

```python
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from MHT_HybridTrainer import HybridTrainer
from MHT_RewardFactory import make_reward
import os
# 1. Load dataset (Wikitext)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# 2. Filter out blank lines (important for Wikitext)
dataset = dataset.filter(lambda x: x["text"] is not None and x["text"].strip() != "")

# 3. Shuffle + select a smaller subset BEFORE tokenization
dataset = dataset.shuffle(seed=42).select(range(200))  # 👈 change 200 to whatever you need

# 4. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = 512

# 5. Preprocess/tokenize
def preprocess(example):
    text = example["text"].strip()
    prompt = f"### Prompt:\n\n```{text}```\n\n### Response:\n\n"
    output = text + tokenizer.eos_token

    # Tokenize combined prompt+output
    enc = tokenizer(
        prompt + output,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    
    # Mask prompt portion in labels
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    labels = enc["input_ids"].copy()
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

    enc["labels"] = labels
    return enc

train_data = dataset.map(preprocess, remove_columns=["text"])

reward_spec = {
    "type": "sentiment",  # Specify the reward type
    "target": "positive"  # Specify the target sentiment
}
# 6. Reward function
reward_fn = make_reward(reward_spec)

# 7. Initialize trainer
trainer = HybridTrainer(
    base_model_name="gpt2",
    tokenizer=tokenizer,
    train_data=train_data,
    reward_fn=reward_fn,
    optimizer_config={"OPTIMIZER": "adamw_8bit"},
    scheduler_config={
        "SCHEDULER": "linear",
        "NUM_TRAINING_STEPS": 20000,
        "NUM_WARMUP_STEPS": 1000,
    },
    lrate=2e-5,
    batch_size=1,  # keep small for smoke test
)

os.system('clear')
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# 8. Build DataLoader
dl = DataLoader(train_data, batch_size=1, shuffle=True)

# 9. Debug: run tiny training with just 2 samples
cnt = 0
for step in dl:
    # Ensure all tensors in the batch are on the same device as the model
    step["input_ids"] = step["input_ids"].to(trainer.device)
    step["attention_mask"] = step["attention_mask"].to(trainer.device)
    step["labels"] = step["labels"].to(trainer.device)

    # SFT step
    loss = trainer.sft_step(step)
    print(f"[{cnt}] SFT loss: {loss:.4f}")

    # Generate the model response for the single example (single response, not a batch)
    response_ids = trainer.respond_to_batch(step["input_ids"])

    # Decode the response from token IDs to text
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)  # Only one response

    # Reconstruct the target text from the labels (where labels are not -100)
    target_ids = step["labels"]  # The labels are the target token IDs
    target_ids = target_ids[target_ids != -100]  # Remove the masked tokens (-100)

    # Decode the target tokens into the target text
    target = tokenizer.decode(target_ids, skip_special_tokens=True)

    print(f"---------------------\nRESPONSE: {response}\nTARGET: {target}\n-------------------------")
    # Compute rewards using the fixed reward_fn reference (response, target as text)
    reward = reward_fn(response, target)

    # PPO step (pass reward as a list for single example)
    trainer.ppo_step(step["input_ids"], response_ids, [reward])  # Wrap reward in a list

    # Debugging output: print responses and rewards
    if trainer.debug:  # Use trainer.debug instead of self.debug
        print(f"[DEBUG] response={response}")
        print(f"[DEBUG] reward={reward}")

    cnt += 1
    if cnt >= 2:  # Only run 2 iterations for the smoke test
        break
```

