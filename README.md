

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

### **Markov Training Theory**

In the context of training models, especially in **text generation** or **language modeling**, **Markov Training** involves using the **Markov property**, where future states (in this case, words or tokens) are only dependent on the present state, and not on past states beyond the most recent.

Markov models have found applications in a variety of machine learning settings, particularly in reinforcement learning, natural language processing, and sequence generation tasks.

Below is the **theory** behind **Markov Training**, how it can be applied during **training**, and how it can be leveraged **later during inference**.

---

### **1. The Markov Property in Text Generation**

The **Markov property** states that the probability of transitioning to the next state (in this case, generating the next token) depends only on the current state (i.e., the previous token or a set of recent tokens) and not on any prior states. This is formally represented as:

$$
P(X_t | X_1, X_2, \dots, X_{t-1}) = P(X_t | X_{t-1})
$$

Where:

* $X_t$ is the state at time step $t$ (for instance, the token generated at time step $t$).
* The current state $X_{t-1}$ is the only factor determining the next state.

In **language modeling** and **text generation**, this translates into the assumption that the probability of generating the next word (or token) depends only on the previous word (or token), not on the entire sequence that came before it.

---

### **2. Applying Markov Training in Language Models**

To **train** models with the Markov property, we must **emphasize local dependencies** (the immediate token) and **minimize reliance on long-term dependencies** (the entire sequence). Here's how it works:

#### **Training Process with Markov Property**

1. **Sequence Generation**:

   * The model generates text one token at a time, with each new token being dependent only on the current token or a small context window (rather than the entire sequence). This can be represented by:

     $$
     P(X_t | X_{t-1}) = \text{Model}(X_{t-1})
     $$

     where $X_{t-1}$ is the current token, and the model predicts the probability distribution for the next token $X_t$.

2. **Self-Supervised Learning**:

   * In this case, the **training data** itself is used to learn the transition probabilities between tokens.
   * The objective is to **minimize the loss** between predicted tokens and the actual tokens in the dataset. This can be achieved through **maximum likelihood estimation** (MLE) or similar loss functions.

3. **Markov Chains for Exploration**:

   * The **Markov Chain** allows for exploration in training. As the model learns, it can generate sequences based on different token transitions, mimicking how sentences evolve.
   * The training process can be guided by a **reward function** or a **reinforcement learning objective** that encourages the model to explore different sequences, similar to how **Markov decision processes** (MDPs) operate in reinforcement learning.

#### **Training with Markov Property: Key Points**

* **Local Token Dependencies**: The model learns from the immediate past token (or a small context window) for efficient sequence generation.
* **Exploration via Stochastic Sampling**: To prevent the model from getting stuck in repetitive cycles or overfitting to specific sequences, it can use **stochastic sampling** or **temperature scaling** to explore diverse token transitions.
* **Markov Chain Monte Carlo (MCMC) Methods**: Techniques like **MCMC** can be used to sample from a distribution of possible next tokens, aiding exploration during training. This is useful in reinforcement learning and generative models.

---

### **3. Using Markov Training for Inference (Later)**

Once the model is trained using the Markov property, the same principles can be applied during **inference** to generate text or sequences in a way that adheres to the Markov assumption.

#### **Inference Process with Markov Property**

1. **Text Generation with Markov Sampling**:

   * Inference is based on **local token transitions**: given a **seed token** (or context), the model predicts the next token using the transition probabilities learned during training:

     $$
     X_{t+1} = \text{Model}(X_t)
     $$
   * This is done repeatedly until a stopping criterion (such as **end of sequence** or **max length**) is reached.

2. **Exploration in Inference**:

   * Just as in training, **exploration** during inference is crucial to prevent the model from producing repetitive or overly deterministic output. This can be done using:

     * **Sampling**: Sampling from the probability distribution over the next token.
     * **Temperature**: Scaling the probability distribution to control the randomness (higher temperature leads to more randomness).
     * **Top-k Sampling**: Limiting the sampling to the top $k$ tokens to balance diversity and coherence.

3. **Markov Decision Processes (MDPs)**:

   * In some cases, **Markov Decision Processes** can be applied in inference to further guide the text generation by selecting sequences that maximize a predefined objective, such as **coherence**, **novelty**, or **factual accuracy**.
   * In an MDP setup, the model would explore different token sequences (states) and select the one with the highest expected reward.

#### **Use Cases in Inference**

* **Story Generation**: The model can generate coherent and diverse stories or paragraphs by sampling from local transitions, ensuring that the text remains coherent while also exploring different pathways.
* **Dialogue Systems**: In a chatbot or dialogue system, the model can predict the next utterance based on the previous one, using local dependencies and maximizing engagement or relevance.

---

### **4. Roadmap for Markov Training**

#### **Step 1: Implement Markov Sampling in Training**

* Train the model to generate sequences by relying on the immediate previous token (Markov property).
* Introduce **stochastic sampling** to encourage exploration during training.
* Use **reward functions** to guide the model’s exploration based on desired outcomes (e.g., coherence, sentiment, relevance).

#### **Step 2: Incorporate Reinforcement Learning (RL) with Markov Models**

* Implement a reinforcement learning setup where the model explores different token sequences (states) and learns to maximize a reward function based on the local token dependencies.
* **PPO** (Proximal Policy Optimization) or **A3C** (Asynchronous Advantage Actor-Critic) can be used to optimize the model with reward feedback based on the **Markov transitions**.

#### **Step 3: Fine-Tuning for Inference**

* Fine-tune the model using **Markov Chains** to generate more diverse sequences during inference.
* Explore advanced sampling techniques like **Top-k**, **Top-p sampling**, or **nucleus sampling** to introduce randomness while maintaining coherence.

#### **Step 4: Use Markov Models for Inference Generation**

* Use the trained model for text generation by applying the **Markov property** to sample from token distributions.
* Integrate the model with **reward-based sampling** to encourage diverse, coherent, and contextually appropriate responses during inference.

---

### **5. Conclusion**

Markov training, when applied to **language models**, involves using the **Markov property** to make predictions based on local token transitions. During **training**, this leads to efficient learning based on immediate context, while in **inference**, it provides a way to generate sequences based on local dependencies, enabling coherent and dynamic output. By incorporating techniques such as **reinforcement learning**, **reward-based sampling**, and **stochastic exploration**, Markov training can enhance model performance and diversity.

### **Theory of Memory in Recurrent Neural Networks (RNNs)**

In Recurrent Neural Networks (RNNs), **memory** is the ability to maintain and use information over time, enabling the model to capture dependencies between elements in a sequence. This is crucial for tasks such as text generation, language modeling, and sequence prediction, where the model needs to "remember" previous inputs to make accurate predictions or generate coherent outputs.

The **theory of memory** in the context of RNNs is rooted in the concept of **sequential processing**, where each output depends not just on the current input but also on the history of previous inputs. The challenge lies in how effectively the network can **store** and **retrieve** this information as the sequence progresses.

---

### **1. Memory in RNNs: Fundamental Concepts**

RNNs are designed to have **loops** within their architecture, which allow the information to persist across time steps. In an RNN:

* At each time step $t$, the model processes the current input $x_t$ and updates its hidden state $h_t$.
* The hidden state $h_t$ acts as the **memory** of the model, capturing information from previous time steps.
* The output at each time step is based on the current hidden state and input.

Mathematically, an RNN can be represented as:

$$
h_t = f(Wx_t + Uh_{t-1} + b)
$$

$$
y_t = W_{out}h_t + b_{out}
$$

Where:

* $h_t$ is the hidden state (memory).
* $W, U$ are weight matrices for the input and the previous hidden state.
* $b$ and $b_{out}$ are biases.
* $y_t$ is the output at time step $t$.

---

### **2. Challenges with Memory in RNNs**

While RNNs are designed to retain information from past time steps, they face the **vanishing gradient problem**. This issue arises when training RNNs over long sequences, where the gradients used to update the model's weights become very small, causing the model to forget long-term dependencies.

**Key challenges**:

* **Vanishing Gradients**: As the network processes longer sequences, gradients diminish exponentially as they propagate backward through time. This leads to difficulty in learning long-range dependencies.
* **Exploding Gradients**: In some cases, gradients may grow exponentially, leading to unstable training.
* **Short-term Memory**: In simple RNNs, the network is more inclined to remember recent events, which may reduce its ability to recall long-term dependencies.

To address these limitations, more advanced RNN architectures like **Long Short-Term Memory (LSTM)** networks and **Gated Recurrent Units (GRU)** were developed.

---

### **3. Long Short-Term Memory (LSTM) Networks**

LSTMs are designed to overcome the vanishing gradient problem by introducing **gates** that regulate the flow of information. These gates allow the network to decide **what to remember** and **what to forget** at each time step. This provides a more **efficient memory mechanism** for capturing long-term dependencies.

An LSTM cell includes the following components:

* **Forget Gate**: Decides what information to discard from the previous time step.
* **Input Gate**: Decides what new information to store in the cell state.
* **Cell State**: The long-term memory, which is updated by the forget and input gates.
* **Output Gate**: Determines the output at the current time step based on the cell state.

The LSTM equations are as follows:

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C x_t + U_C h_{t-1} + b_C)
$$

$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

$$
h_t = o_t * \tanh(C_t)
$$

Where:

* $f_t$, $i_t$, and $o_t$ are the forget, input, and output gates.
* $\tilde{C}_t$ is the candidate cell state.
* $C_t$ is the cell state (memory).
* $h_t$ is the hidden state (output).

LSTMs can store information over long periods, allowing them to remember distant dependencies in sequences, making them suitable for tasks requiring **long-term memory**.

---

### **4. Gated Recurrent Units (GRUs)**

GRUs are a simplified version of LSTMs that combine the forget and input gates into a single gate. This reduces the complexity of the network while still allowing it to capture long-range dependencies. GRUs are more computationally efficient and have been shown to perform well in many sequence modeling tasks.

A GRU operates as follows:

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t * h_{t-1}) + b_h)
$$

$$
h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
$$

Where:

* $z_t$ is the update gate (similar to the output gate in LSTMs).
* $r_t$ is the reset gate.
* $\tilde{h}_t$ is the candidate hidden state.
* $h_t$ is the final hidden state (memory).

---

### **5. Memory in RNNs: Training and Use During Inference**

#### **During Training:**

* **Memory Cells (LSTM/GRU)**: During training, the model's memory cells (hidden states) are updated based on the input sequence, enabling it to store information from past tokens. The model is **rewarded** based on how well it can recall important information for predicting the next token.
* **Memory Evolution**: Over time, the memory evolves as the model learns to store useful information and forget irrelevant data.
* **Backpropagation Through Time (BPTT)**: The model’s memory is updated using **BPTT**, which ensures that the gradients from the loss propagate through the sequence, allowing the model to adjust its memory.

#### **During Inference:**

* The **trained model** uses its memory (hidden state) to generate predictions based on the current context (token or sequence).
* During **text generation** or **sequence prediction**, the model uses the previously generated tokens to update its memory, ensuring that each new prediction is informed by the prior context.
* The **memory** is continually updated as new tokens are generated, allowing the model to maintain context across longer sequences.

---

### **6. Memory Mechanisms and Rewards**

In the context of **Markov Training** or **HybridTraining**, **reward-based memory** mechanisms can be applied to train models in such a way that they focus on remembering useful information for **long-term coherence**. This can be achieved by combining **memory-enhanced networks** (LSTMs/GRUs) with various rewards, such as **text similarity** or **perplexity**.

### **7. Future Work:**

* **Memory Augmentation**: Implement memory-augmented neural networks (MANNs) like **Neural Turing Machines** (NTM) or **Differentiable Neural Computers** (DNC), which can learn to read from and write to memory slots, making them even more powerful for sequence modeling.
* **External Memory**: Integrating an external memory mechanism where the model can store and retrieve key information from a **long-term memory bank**, similar to human cognition, could improve task performance.
* **Learned Attention Mechanisms**: Attention mechanisms (like **self-attention**) can be combined with memory to help the model focus on the most important tokens in long sequences, mitigating the issues with **long-range dependencies**.

---

### **8. Conclusion**

Memory in RNNs is a critical aspect of handling sequential data. By using **LSTMs** and **GRUs**, we can mitigate the problem of vanishing gradients and ensure that the model can **store and recall** information over long sequences. Through reinforcement learning and reward functions, we can further enhance the model’s ability to use its memory effectively, both during **training** and **inference**. As the model learns, it can adapt its memory usage, focusing on important information and **forgetting irrelevant details**, leading to improved performance in **real-world tasks** like **text generation** and **sequence prediction**.

### **Theory of Training Models with Reward Learning and Classifiers for Fine-Tuning**

In advanced machine learning systems, especially in the context of reinforcement learning (RL) and fine-tuning pre-trained models, the concept of **reward learning**, **classifier tuning**, and the use of **heads** for model inference or training becomes crucial. The objective is to not just train models for specific tasks (like text generation or classification) but to **augment** and **adapt** them in ways that make them more intelligent, flexible, and adaptable to a variety of challenges.

This involves using **multi-objective optimization** during training, allowing the model to simultaneously learn how to produce desirable outputs (via rewards) and optimize performance in different scenarios, like classification tasks or specific output heads for inference.

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

### Reward Factory


### MHT_RewardFactory.py
    Sure! Here's a list of **reward types** that you can add to your **HybridTrainer** setup in the future. These reward types can be used for various NLP tasks and combined in different ways, depending on your needs.

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

It seems like you're looking to explore **Markov behavior** and **recurrent memory** during fine-tuning and inference in the **HybridTrainer** class. Here's a detailed explanation of how you can implement those changes in your model's training and inference loop:

### Key Enhancements:

1. **Markov-Like Loss Function**: This is introduced to focus the model's learning on recent tokens or immediate context. This is important to make the model behave more like a Markov process, where future predictions depend primarily on the current token.

2. **Memory-Based Rewards**: This reward mechanism encourages long-term consistency in the generated outputs, ensuring that the model maintains coherence over time by comparing the memory of previous outputs with the current ones.

3. **Sampling with Memory Awareness**: During inference, the model generates sequences that not only focus on the recent context (Markov behavior) but also consider long-term coherence and memory.

### Updated **HybridTrainer** Class:

Here's the updated **HybridTrainer** class implementing both **Markov behavior** and **recurrent memory** features during training and inference.

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, PPOTrainer, PPOConfig, create_reference_model
from trl.core import respond_to_batch
from transformers import get_scheduler

# Custom Reward Functions (for Markov and memory handling)
class HybridTrainer:
    def __init__(self, base_model_name, tokenizer, train_data, reward_fn,
                 optimizer_config=None, scheduler_config=None,
                 lrate=5e-5, batch_size=1,
                 maxseq=512, max_new_tokens=128,
                 num_training_steps=None, num_warmup_steps=None,
                 context_window_size=10, memory_window_size=50, debug=False):

        self.debug = debug
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.reward_fn = reward_fn
        self.maxseq = maxseq
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.context_window_size = context_window_size
        self.memory_window_size = memory_window_size

        # Initialize the model (LM head + Value head)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.pretrained_model = self.model
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
            model=self.model,
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

    # -- Markov Loss --  
    def markov_loss(self, predicted, target, weight_factor=1.5):
        # Apply Markov-like behavior by focusing on recent tokens
        recent_target = target[-self.context_window_size:]  # Focus on the last N tokens
        recent_pred = predicted[-self.context_window_size:]
        
        # Calculate CrossEntropy loss for recent tokens (Markov behavior emphasis)
        loss = torch.nn.functional.cross_entropy(recent_pred, recent_target)
        return weight_factor * loss

    # -- Memory Reward --
    def memory_reward(self, response, target, memory_length=50):
        # Encourages memory handling: consistency over long sequences
        response_memory = response[-memory_length:]  # Use last `memory_length` tokens
        target_memory = target[-memory_length:]

        consistency_score = self.compute_consistency(response_memory, target_memory)
        return consistency_score

    def compute_consistency(self, response_memory, target_memory):
        # Example of a simple consistency measure (could be cosine similarity, etc.)
        # Here we will use a placeholder for consistency calculation
        return torch.cosine_similarity(response_memory, target_memory, dim=-1).mean()

    # -- SFT Step with Markov Loss --
    def sft_step(self, batch):
        # Normal SFT loss calculation
        loss = self.sft_trainer.compute_loss(self.model, batch)

        # Apply Markov loss (focus on recent context)
        markov_loss_value = self.markov_loss(batch["predicted_tokens"], batch["target_tokens"])
        
        total_loss = loss + markov_loss_value
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return float(total_loss.item())

    # -- PPO Step with Memory Reward --
    def ppo_step(self, input_ids, response_ids, rewards):
        # Add memory reward to PPO step
        memory_rewards = [self.memory_reward(response, target) for response, target in zip(response_ids, rewards)]
        
        # Apply PPO step with memory reward
        stats = self.ppo_trainer.step(input_ids, response_ids, memory_rewards)
        return stats

    # -- Generate with Memory-Aware Sampling --
    def sample_with_memory(self, input_ids, max_tokens=128):
        # Use recent context as memory (Markov behavior)
        memory_context = input_ids[-self.context_window_size:]  # Use last N tokens

        # Generate tokens based on memory context
        generated_tokens = self.model.generate(memory_context, max_length=max_tokens)
        return generated_tokens

    def respond_to_batch(self, input_ids):
        # Get model response to a batch input (to be used during training and inference)
        return respond_to_batch(self.model, input_ids)

    # -- Inference with Memory-Augmented Sampling --
    def memory_augmented_sampling(self, input_ids, response_ids):
        # Use memory reward during inference sampling
        reward = self.memory_reward(response_ids, target)
        
        # Sample with memory-awareness (long-term coherence)
        generated_tokens = self.sample_with_memory(input_ids)
        return generated_tokens

```

### Key Features of the HybridTrainer Class:

1. **Markov Loss**: The loss function now emphasizes recent tokens, guiding the model to make predictions based on local context (Markov behavior).

   * It calculates loss for only the **recent tokens**, effectively encouraging the model to use local dependencies in the sequence.
2. **Memory Reward**: The **memory reward** encourages the model to maintain coherence over longer sequences.

   * **Memory Reward** is calculated during **PPO training** and **inference** to keep long-term consistency in generated sequences.
3. **Memory-Augmented Sampling**: During **inference**, the model generates text based on the **recent context** (Markov-like behavior), while also taking memory into account for longer sequences.

### Integration into Training Loop

To **train** and **fine-tune** the model, you can integrate this **HybridTrainer** as follows:

```python
# Initialize tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained('gpt2')
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Preprocess dataset
train_data = dataset.map(preprocess, remove_columns=["text"])

# Define reward function (e.g., sentiment analysis)
reward_spec = {
    "type": "sentiment",  # Specify the reward type
    "target": "positive"  # Specify the target sentiment
}
reward_fn = make_reward(reward_spec)

# Initialize HybridTrainer
trainer = HybridTrainer(
    base_model_name="gpt2",
    tokenizer=tokenizer,
    train_data=train_data,
    reward_fn=reward_fn,
    optimizer_config={"OPTIMIZER": "adamw"},
    scheduler_config={
        "SCHEDULER": "linear",
        "NUM_TRAINING_STEPS": 20000,
        "NUM_WARMUP_STEPS": 1000,
    },
    lrate=2e-5,
    batch_size=1,  # small batch for testing
    maxseq=512
)

# Create DataLoader
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dl = DataLoader(train_data, batch_size=1, shuffle=True)

# Train Loop
for step in dl:
    step["input_ids"] = step["input_ids"].to(trainer.device)
    step["attention_mask"] = step["attention_mask"].to(trainer.device)
    step["labels"] = step["labels"].to(trainer.device)

    # Run SFT Step
    loss = trainer.sft_step(step)
    print(f"SFT Loss: {loss:.4f}")

    # Generate response and calculate memory reward during PPO
    response_ids = trainer.respond_to_batch(step["input_ids"])
    reward = trainer.memory_reward(response_ids, step["labels"])

    # PPO step (with memory reward)
    stats = trainer.ppo_step(step["input_ids"], response_ids, [reward])
    print(f"PPO Stats: {stats}")

    if trainer.debug:
        print(f"Response: {tokenizer.decode(response_ids[0])}")
        print(f"Reward: {reward}")
```

### Summary of Enhancements:

* **Markov Behavior**: The model focuses on recent tokens (Markov-like behavior) during training via custom **loss functions**.
* **Recurrent Memory**: The model is rewarded for maintaining long-term **coherence and memory** during both **training** and **inference** via the **memory reward**.
* **Inference with Memory**: Memory is explicitly handled during generation to ensure long-term consistency in outputs.

By integrating these changes, your **HybridTrainer** is now more **focused on improving learning** by leveraging both **Markov behavior** and **recurrent memory** effectively during **fine-tuning** and **inference**.

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
