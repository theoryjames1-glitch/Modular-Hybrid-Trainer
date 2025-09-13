### **Theory of Modular HybridTrainer**

The **HybridTrainer** is an architecture that combines **supervised learning** and **reinforcement learning (RL)** in a flexible, modular framework. The goal is to train an AI model using a combination of different learning paradigms and reward signals, making it adaptable to various tasks and environments. By incorporating multiple **reward functions** (such as **sentiment analysis**, **truthfulness assessment**, **fluency metrics**, and more), the **HybridTrainer** can guide the model to generate responses that adhere to specific goals or qualities, like **positive sentiment**, **truthfulness**, **coherence**, etc.

### **Key Concepts for Modular HybridTrainer**:

1. **Modular Design**: Each component (optimizer, scheduler, reward function, etc.) is a **plug-in module** that can be replaced or modified easily based on the task. This modularity allows flexibility in designing different training scenarios.

2. **Supervised + Reinforcement Learning (RL)**:

   * **Supervised learning** helps the model learn from labeled data.
   * **Reinforcement learning** provides dynamic feedback based on the model's performance in generating outputs in real-time, helping the model adjust and improve over time.

3. **Customizable Reward Functions**:

   * The HybridTrainer integrates multiple **reward functions**, such as **sentiment analysis**, **truthfulness**, **perplexity**, **toxicity**, and **BLEU/ROUGE scores**. These rewards are used in the **RL** step to evaluate the quality of the model’s responses.
   * **Modular reward functions** are easy to add or remove as needed. This flexibility enables the model to learn based on various goals (e.g., truthfulness, positive sentiment, coherence, etc.).

4. **Memory & Adaptation**:

   * A key feature of the **HybridTrainer** is the use of an **adaptive memory** system. This allows the model to "remember" past interactions, rewards, and learned patterns. It classifies the experiences into **easy, medium, and hard** categories, allowing the model to focus more on hard or underperforming tasks.
   * **Markov chains** or **episodic memory** systems can be used to keep track of the model’s past interactions, providing long-term learning from previous experiences.

---

### **Roadmap for Implementing the Modular HybridTrainer**

Here’s a structured **roadmap** for how we can integrate the modular components into the **HybridTrainer** system.

---

### **1. Initial Design of the HybridTrainer System**

* **Objective**: Build a flexible architecture that allows for easy swapping of components (optimizers, schedulers, reward functions, etc.) depending on the task.

#### **Tasks**:

* Design the overall structure of the HybridTrainer, ensuring that it has clear interfaces for different modules:

  * **Supervised learning module** (e.g., for fine-tuning with labeled data).
  * **Reinforcement learning module** (e.g., for using PPO or other RL algorithms for dynamic feedback).
  * **Reward functions**: Create a **modular reward factory** to support various reward types (e.g., truthfulness, sentiment, perplexity, BLEU, etc.).

---

### **2. Integrating Supervised Learning Step**

* **Objective**: Train the model with labeled data, applying supervised learning first. This step ensures the model learns basic language patterns.

#### **Tasks**:

* **Step 1: Dataset Preparation**: Load training datasets in the desired format (e.g., JSON, CSV, or text files).
* **Step 2: Model Selection**: Choose a pretrained model (e.g., GPT, BERT) and fine-tune it using the labeled dataset.
* **Step 3: Supervised Learning Module**: Implement the supervised learning step, where the model is trained using traditional **cross-entropy loss** or other appropriate loss functions.

---

### **3. Integrating Reinforcement Learning (RL)**

* **Objective**: Add a reinforcement learning step after supervised learning, where the model is further fine-tuned using a custom reward signal (e.g., truthfulness, sentiment, etc.).

#### **Tasks**:

* **Step 1: RL Algorithm Selection**: Implement an RL algorithm, such as **Proximal Policy Optimization (PPO)** or **A3C**, which will use feedback from reward functions to guide training.
* **Step 2: Reward Function Integration**: Integrate **modular reward functions** (e.g., sentiment analysis, truthfulness NLI, perplexity) into the RL loop. The reward function will give feedback based on how well the model's output aligns with the desired objective.
* **Step 3: Training Loop**: Create the training loop that alternates between **supervised learning** and **reinforcement learning**:

  1. **Supervised Learning**: Pretrain the model with labeled data.
  2. **Reinforcement Learning**: Fine-tune the model using rewards (from sentiment, truthfulness, etc.) as feedback.

---

### **4. Modular Reward Factory**

* **Objective**: Design a modular reward factory that can plug in different reward functions based on the task at hand. Each reward function will give feedback about different aspects of the generated text (e.g., sentiment, truthfulness, fluency).

#### **Tasks**:

* **Step 1: Define Reward Types**: Define several reward functions like **Cosine Similarity (for semantic similarity)**, **Perplexity (for fluency)**, **Truthfulness (via NLI)**, and **Sentiment (via Hugging Face)**.
* **Step 2: Reward Function Factory**: Implement a reward factory that can dynamically select and combine these reward functions based on the task configuration.
* **Step 3: Post-Processing Rewards**: Add the ability to **post-process** rewards with techniques like **clipping**, **scaling**, or **EMA normalization** to stabilize the learning process.

---

### **5. Adaptive Memory and Markov Chain Integration**

* **Objective**: Add a memory system that allows the model to remember past experiences (prompts, responses, losses) and adapt over time.

#### **Tasks**:

* **Step 1: Memory System**: Implement a system that stores previous experiences in a memory buffer, classifying them into **easy**, **medium**, and **hard** categories.
* **Step 2: Markov Chains**: Use Markov Chains or other memory techniques to manage the **sequence of events** and their probabilities. This allows the model to make decisions based on historical data.
* **Step 3: Reinforcement Learning with Memory**: Allow the RL module to take into account both **immediate rewards** and **past experiences** when updating the model. This helps the model improve over time based on long-term feedback.

---

### **6. Fine-Tuning and Testing**

* **Objective**: After the model has gone through the **supervised** and **reinforcement learning** steps, test it with new data to ensure it performs well across various scenarios.

#### **Tasks**:

* **Step 1: Fine-Tuning**: Fine-tune the model using **hyperparameter tuning** and **validation** techniques to ensure it is learning optimally.
* **Step 2: Evaluation**: Evaluate the model on new, unseen data using a variety of metrics (e.g., BLEU, ROUGE, F1 score).
* **Step 3: Benchmarking**: Compare the performance of the **HybridTrainer** against other baselines (e.g., standard supervised learning or RL-based models).

---

### **7. Deployment and Monitoring**

* **Objective**: After testing and fine-tuning, deploy the model and continuously monitor its performance in real-world settings.

#### **Tasks**:

* **Step 1: Deployment**: Deploy the trained model to a server or application, where it can be accessed for real-time inference.
* **Step 2: Continuous Learning**: Implement a feedback loop that allows the model to **adapt** based on new incoming data (via supervised or RL methods).
* **Step 3: Monitoring**: Continuously monitor the model’s performance using evaluation metrics, ensuring that the model is improving over time.

---

### **Modular HybridTrainer Roadmap Summary**:

1. **Design the HybridTrainer architecture** to support both supervised and RL steps.
2. **Integrate modular reward functions** like sentiment, truthfulness, fluency, etc.
3. **Implement memory systems** (e.g., Markov Chains) to store past experiences and adapt over time.
4. **Train with a combination of supervised learning** for foundational knowledge and **reinforcement learning** for real-time feedback.
5. **Fine-tune and evaluate** the system based on various metrics to ensure optimal performance.
6. **Deploy and monitor** the model in real-world settings, enabling continuous learning and improvement.

By following this roadmap, you'll create a **modular and scalable** system that can adapt to a wide range of tasks, with flexibility to plug in new reward functions or learning paradigms as necessary.

---

### **1. HybridTrainer Architecture**

We'll build a basic class, `HybridTrainer`, with methods for setting up supervised and reinforcement learning, modular reward functions, and reward aggregation.

#### **1.1: Setup HybridTrainer Class**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.optim import AdamW
from typing import List, Dict, Optional

class HybridTrainer:
    def __init__(self, model_name: str, train_data: torch.utils.data.Dataset, reward_fn: Optional[callable] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler] = None):
        """
        Initialize HybridTrainer with modular components.
        """
        self.model_name = model_name
        self.train_data = train_data  # dataset for supervised learning
        self.reward_fn = reward_fn    # callable reward function for reinforcement learning
        self.optimizer = optimizer if optimizer else AdamW(self.model.parameters(), lr=1e-5)  # default optimizer
        self.scheduler = scheduler    # Learning rate scheduler
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_model(self):
        """
        Load model from Hugging Face model hub or a local checkpoint.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.train()  # Set the model to training mode
        return model

    def train_supervised(self, epochs: int = 1, batch_size: int = 8):
        """
        Train the model in supervised learning mode (using labeled data).
        """
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            optimizers=(self.optimizer, self.scheduler)
        )

        trainer.train()

    def train_reinforcement(self, epochs: int = 1, batch_size: int = 8):
        """
        Train the model using reinforcement learning (with rewards).
        """
        # Placeholder for RL loop. You can plug in your RL algorithm here (e.g., PPO).
        for epoch in range(epochs):
            for step, batch in enumerate(self.train_data):
                inputs = self.tokenizer(batch['input'], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                targets = self.tokenizer(batch['output'], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                
                # Forward pass (get model's predictions)
                outputs = self.model(**inputs, labels=targets['input_ids'])
                loss = outputs.loss

                # Compute reward from the reward function
                reward = self.reward_fn(outputs, batch)
                reward_loss = loss * reward  # Adjust loss by the reward

                # Backpropagate the loss
                self.optimizer.zero_grad()
                reward_loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                if step % 10 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss {reward_loss.item()}, Reward {reward}")

    def run_training(self, supervised_epochs: int = 1, rl_epochs: int = 1, batch_size: int = 8):
        """
        Run both supervised and reinforcement learning training.
        """
        print("Starting supervised training...")
        self.train_supervised(epochs=supervised_epochs, batch_size=batch_size)

        print("Starting reinforcement learning...")
        self.train_reinforcement(epochs=rl_epochs, batch_size=batch_size)

```

---

### **2. Reward Functions Integration**

We'll integrate the reward functions (like **sentiment**, **truthfulness**, **perplexity**) that we created earlier. For simplicity, let's assume that rewards will be calculated based on output predictions from the model.

We'll add a **modular reward factory** that will allow us to plug in different reward functions for RL-based fine-tuning.

```python
# rewards.py

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util as st_util

class Reward:
    """Base interface for rewards."""
    def __call__(self, model_output, batch):
        raise NotImplementedError

class SentimentReward(Reward):
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english", target="positive"):
        self.model = pipeline("sentiment-analysis", model=model)
        self.target = target

    def __call__(self, model_output, batch):
        prediction = self.model(model_output["generated_text"])[0]
        label = prediction["label"].lower()
        score = prediction["score"]
        if self.target in label:
            return score
        return 1.0 - score

class TruthfulnessReward(Reward):
    def __init__(self, model="roberta-large-mnli"):
        self.model = pipeline("text-classification", model=model, return_all_scores=True)

    def __call__(self, model_output, batch):
        response = model_output["generated_text"]
        target = batch["target"]  # The reference truth or target string
        result = self.model({"text": target, "text_pair": response})[0]
        scores = {d["label"].lower(): float(d["score"]) for d in result}
        p_e = scores.get("entailment", 0.0)
        p_c = scores.get("contradiction", 0.0)
        return max(0.0, min(1.0, 0.5 * (p_e - p_c) + 0.5))

```

---

### **3. Modular Reward Factory**

Create a **reward factory** to generate rewards dynamically based on JSON configuration or user input:

```python
def make_reward(spec: Dict[str, Any]) -> Reward:
    t = spec.get("type", "").lower()
    if t == "sentiment":
        return SentimentReward(target=spec.get("target", "positive"))
    if t == "truthfulness":
        return TruthfulnessReward(model=spec.get("model", "roberta-large-mnli"))
    # Add other reward functions here...
    raise ValueError(f"Unknown reward type: {t}")
```

---

### **4. Example JSON Configuration**

Here’s how you can configure and use **modular rewards** in a JSON format for the **HybridTrainer**:

#### **Example JSON Configurations**

**A) Sentiment-based reward:**

```json
{
  "REWARD": {
    "type": "sentiment",
    "target": "positive",
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

**B) Truthfulness-based reward:**

```json
{
  "REWARD": {
    "type": "truthfulness",
    "model": "roberta-large-mnli",
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

---

### **5. Connecting Everything Together**

Now we can connect all modules and integrate the training process:

```python
# main.py

from rewards import make_reward
from datasets import load_dataset

# Example: Loading training data (adjust to your dataset)
train_data = load_dataset('path_to_your_dataset')

# Load reward function (this is an example, the actual reward function can be configured in a JSON file)
reward_spec = {"type": "truthfulness", "model": "roberta-large-mnli"}
reward_fn = make_reward(reward_spec)

# Instantiate HybridTrainer with dataset and reward function
trainer = HybridTrainer(
    model_name="gpt2",
    train_data=train_data["train"],  # Using the dataset
    reward_fn=reward_fn,  # Plug-in reward function
)

# Run training
trainer.run_training(supervised_epochs=1, rl_epochs=3, batch_size=4)
```

---

### **6. Conclusion**

The **HybridTrainer** system is now modular and can easily integrate various learning paradigms (supervised and RL) and reward functions (sentiment, truthfulness, etc.). You can swap out or add new reward types based on the task, and the entire process is flexible and extensible.

### **Next Steps**:

* Expand the **reward factory** with more custom reward functions.
* Add **more sophisticated RL algorithms** (e.g., **PPO**, **A3C**).
* **Test the trainer** on various datasets to fine-tune performance.

---

Yes, **Markov Chains** can be an interesting way to enhance **Adaptive Memory** in machine learning models, especially for tasks like **sequential memory retention** and **pattern prediction**.

In the context of **Adaptive Memory**, Markov Chains could be used to:

1. **Track sequences** of prompts, responses, and actions, as the model learns which patterns and transitions (from one prompt/response to another) are more common or successful.
2. **Predict and reinforce important memory transitions**: the system could prioritize revisiting states (prompts/responses) that are more likely to lead to successful outcomes in the future, based on historical data.
3. **Facilitate decision-making**: by modeling the model's decisions (and the states they transition to) as a Markov process, you could allow the system to adapt its behavior based on the "most probable" next states that lead to higher rewards or learning success.

Let’s break this down:

### Key Concepts

1. **Markov Chains**:
   A Markov Chain models a system where the probability of moving to the next state depends only on the current state (the **Markov property**). This means the future state is independent of past states beyond the current one.

   * In this case, states could represent **sequences of prompts** or **outputs** (responses).
   * The transitions between these states could be driven by model **losses** or **rewards** (e.g., probability of producing correct responses, or achieving high reward in RL tasks).

2. **Memory States**:
   Each **state** in the Markov Chain could represent a **sequence of training data** or **interaction** — for example:

   * The **current prompt** the model is given.
   * The **model’s response** (or action) in that state.
   * The **reward** or **loss** received in that state.

3. **Transition Probabilities**:
   The **transition probability** between two states could represent how likely it is to move from one **prompt-response pair** to another, given that the model has previously encountered that pair in training. The likelihood of transitions would be based on the **rewards**, **losses**, or **successes** from the model’s training interactions.

4. **Memory Reinforcement**:
   Using a Markov Chain, we could reinforce the memory transitions that **lead to better outcomes** or **higher rewards** (e.g., the model will more often "revisit" prompts or responses that resulted in high rewards, or were otherwise successful).

---

### How Markov Chains Could Work in Adaptive Memory

1. **Markov Chain Memory Representation**:

   * Create a transition matrix, where rows represent **states** (sequences of prompt-response pairs) and columns represent **future states**.
   * Each cell in the matrix will represent a transition probability (how likely it is that the current state will lead to the next state).
   * As the model interacts with new prompts and generates responses, update these transitions based on the reward feedback (or the success/failure of that response).

2. **Adaptation to Rewards**:

   * Markov Chains inherently work by modeling state transitions. If a certain prompt-response pair (state) leads to a **positive reward**, you increase the transition probability to that state.
   * If the model’s response is poor, you **decrease** the transition probability, signaling that this state should be avoided in future transitions.
   * Over time, the memory will **adapt** to keep the states (prompts and responses) that lead to **higher rewards** and discard the ones that lead to failure.

3. **Reinforcement Learning via Markov Chains**:

   * When using **reinforcement learning (RL)**, the **transition matrix** can serve as a guide for the RL agent, helping it to focus on state transitions that have higher expected rewards.
   * The **reward function** could then be used to modify the transition probabilities, gradually reinforcing **desirable state transitions** and avoiding **undesirable ones**.

---

### Example of Markov Chain Integration in Adaptive Memory

Here’s how we can **integrate Markov Chains** into the `AdaptiveMemory` class. We'll update it to track **state transitions** between prompts, responses, and rewards, adjusting the transition probabilities over time.

```python
import numpy as np
from collections import defaultdict

class MarkovChainAdaptiveMemory:
    def __init__(self, capacity=20000, state_size=1024, alpha=0.1, gamma=0.9):
        """
        Adaptive Memory using Markov Chains to reinforce successful state transitions.
        
        Args:
        - capacity: Max memory size
        - state_size: Size of the state space (prompt-response pairs)
        - alpha: Learning rate (controls how quickly memory updates)
        - gamma: Discount factor for future rewards
        """
        self.capacity = capacity
        self.state_size = state_size
        self.alpha = alpha
        self.gamma = gamma
        
        # Transition matrix (states x states), initialized to zero
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.state_rewards = defaultdict(float)  # Reward for each state
        
        # Memory tracking (prompt, response pairs)
        self.state_history = []
        self.memory = []

    def _get_state_key(self, prompt, response):
        """
        Convert prompt and response into a hashable state key.
        """
        return (prompt, response)

    def add_to_memory(self, prompt, response, reward):
        """
        Add experience to the memory.
        """
        state = self._get_state_key(prompt, response)
        self.memory.append((state, reward))
        self.state_rewards[state] = reward

        if len(self.memory) > self.capacity:
            # Remove the oldest memory entry
            self.memory.pop(0)

    def update_transition_matrix(self):
        """
        Update the transition matrix based on the current state and reward.
        """
        for i in range(1, len(self.memory)):
            prev_state, prev_reward = self.memory[i-1]
            curr_state, curr_reward = self.memory[i]

            # Increment transition probability for moving from prev_state to curr_state
            self.transition_matrix[prev_state][curr_state] += self.alpha * (prev_reward + self.gamma * curr_reward)

    def get_next_state(self, current_state):
        """
        Given the current state, return the next most likely state based on transition probabilities.
        """
        if current_state not in self.transition_matrix:
            return None
        
        next_state_probabilities = self.transition_matrix[current_state]
        max_prob_state = max(next_state_probabilities, key=next_state_probabilities.get, default=None)
        return max_prob_state

    def sample_memory(self, k=8):
        """
        Sample k states from memory, prioritizing those with higher rewards.
        """
        sorted_states = sorted(self.memory, key=lambda x: self.state_rewards[x[0]], reverse=True)
        return sorted_states[:k]

    def get_transition_probabilities(self):
        """
        Return the transition matrix probabilities for inspection.
        """
        return self.transition_matrix
```

### How It Works:

1. **Transition Matrix**:
   We store the transition probabilities between **prompt-response pairs**. When the model transitions from one state (prompt-response) to another, the matrix is updated with the reward received.

2. **State Rewards**:
   Each state (prompt-response pair) is assigned a **reward** (from SL or RL). The reward is stored and used to adjust transition probabilities.

3. **State Sampling**:
   The model samples from the most successful **prompt-response transitions** based on rewards. This ensures that the model **revisits successful states** for reinforcement learning.

4. **Spaced Repetition**:
   By using **transition probabilities**, you can implement **spaced repetition**. States that led to **high rewards** are **more likely to be revisited**, while those leading to failure are given less attention.

---

### Using Markov Chains for Adaptive Memory in RL & SL

* **For Supervised Learning (SFT)**: You can use the **transition matrix** to prioritize states (prompt-response pairs) that are important to remember, allowing for **targeted rehearsal**.

* **For Reinforcement Learning (PPO)**: Use the **state transitions** and **reward model** to guide the policy updates. Transitions with higher rewards can be **revisited** during policy updates to reinforce behaviors that lead to better outcomes.

* **Continual Memory**: As the model learns, the transition matrix gets updated, and the model continuously adapts its behavior based on **historical successes**.

---

### Benefits of Markov Chains in Adaptive Memory:

* **Reinforcement of Successful States**: By tracking transitions, we can prioritize **rewarding sequences** and help the model **revisit** them.
* **State Prediction**: The model will naturally start predicting which prompts and responses are more likely to lead to success, speeding up learning.
* **Efficient Memory**: We store a manageable **transition matrix** instead of storing every past interaction, which helps with memory efficiency.

---

### Next Steps:

* **Fine-tune Transition Updates**: Adjust the transition matrix updating mechanism, such as introducing decay for old transitions.
* **Rehearsal Strategies**: Use the transition matrix to **selectively replay** difficult states (those that led to poor rewards or low success).
* **Integration with RL & SL**: The `MarkovChainAdaptiveMemory` can now guide both **Supervised Learning** and **Reinforcement Learning** in your Hybrid Trainer.

---

It looks like you have a very solid approach to incorporating **BERTScore** and **Perplexity** into the reward system, and making them **optional and cached** for performance is a smart idea. The modular way of adding these rewards ensures that you can expand or customize your reward function pipeline later. Let's walk through the steps to integrate these rewards with your **Hybrid Trainer**.

Here’s what we’ll do:

### 1) **Integrating BERTScore and Perplexity Rewards into the `HybridTrainer`**

You want to incorporate these two rewards (and possibly more in the future) into the **reward factory** while keeping the training flow modular. We can simply call these rewards as part of the **reward calculation** after the model generates its responses.

### 2) **Modular Reward Integration**:

* **BERTScore**: Measures semantic similarity between the model's response and the target response.
* **Perplexity**: Measures how well the model's response fits within a pre-trained language model's probability distribution (lower perplexity = better response).

### **Step-by-step code for HybridTrainer**

Let’s update the **HybridTrainer** to use the new reward system.

### 1. **HybridTrainer Class** (Updated with reward integration)

```python
class HybridTrainer:
    def __init__(self, model, tokenizer, train_data, reward_fn, 
                 lr_sft=5e-5, lr_ppo=1e-5, batch_size=4, maxseq=1024, 
                 max_new_tokens=128, sft_steps_per_cycle=1, ppo_steps_per_cycle=1):
        # Initialize basic components
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.reward_fn = reward_fn
        self.lr_sft = lr_sft
        self.lr_ppo = lr_ppo
        self.batch_size = batch_size
        self.maxseq = maxseq
        self.max_new_tokens = max_new_tokens
        self.sft_steps_per_cycle = sft_steps_per_cycle
        self.ppo_steps_per_cycle = ppo_steps_per_cycle

        # Initialize the optimizers for SFT and PPO
        self.opt_sft = AdamW(model.parameters(), lr=lr_sft)
        self.ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, learning_rate=lr_ppo)

        # To control stopping criteria
        self.stopping_criteria = None

    def sft_step(self, batch):
        """
        Perform a Supervised Learning (SFT) step.
        """
        # Process the batch and calculate supervised loss
        inputs = self.tokenizer(batch['prompt'], truncation=True, padding=True, max_length=self.maxseq, return_tensors="pt")
        labels = self.tokenizer(batch['output'], truncation=True, padding=True, max_length=self.maxseq, return_tensors="pt")["input_ids"]
        
        # Move inputs and labels to the correct device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        labels = labels.to(self.model.device)

        outputs = self.model(input_ids=inputs["input_ids"], labels=labels)
        loss = outputs.loss

        # Backpropagate the loss
        self.opt_sft.zero_grad()
        loss.backward()
        self.opt_sft.step()

        return loss.item()

    def ppo_step(self, batch):
        """
        Perform a Reinforcement Learning (PPO) step.
        """
        # Collect responses from the model based on the batch
        queries, responses = self.collect_responses(batch)

        # Calculate rewards using the reward_fn
        rewards = [self.reward_fn(r) for r in responses]

        # Perform PPO update
        self.ppo_trainer.step(queries, responses, rewards)

        return sum(rewards) / len(rewards)

    def collect_responses(self, batch):
        """
        Collect responses from the model for the given batch.
        """
        queries = []
        responses = []

        for item in batch:
            prompt = item['prompt']
            target = item['output']
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.maxseq).to(self.model.device)
            
            # Generate a response based on the prompt
            response_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

            queries.append(inputs['input_ids'])
            responses.append(response)

        return queries, responses

    def train(self, epochs=1):
        """
        Train the model alternating between SFT and PPO steps.
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Shuffle training data each epoch
            train_data_shuffled = self.train_data.sample(frac=1).reset_index(drop=True)

            # Training loop
            for step, batch in enumerate(train_data_shuffled.iterrows()):
                batch = batch[1]  # Unwrap to get the actual row of data
                
                # Step 1: Supervised Learning step
                sft_loss = self.sft_step(batch)
                print(f"Step {step} | SFT loss: {sft_loss:.4f}")

                # Step 2: Reinforcement Learning step
                ppo_reward = self.ppo_step(batch)
                print(f"Step {step} | PPO reward: {ppo_reward:.4f}")
                
                # Optional: Save model after each cycle or periodically
                if step % 100 == 0:
                    self.model.save_pretrained("path_to_save_model")
                    self.tokenizer.save_pretrained("path_to_save_model")
                    print(f"Model saved at step {step}")

```

### 2. **Incorporating Reward Classes**

Now, in your reward factory (`rewards.py`), you can add the **BERTScore** and **Perplexity** reward classes. Here’s a simple example of how to use them inside the `HybridTrainer`.

### 3. **Example Config for Hybrid Reward System**

You can now configure a **weighted reward system** for **BERTScore** and **Perplexity** in your JSON:

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "bertscore", "model_type": "microsoft/deberta-xlarge-mnli", "rescale_with_baseline": true, "weight": 0.6},
      {"type": "perplexity", "model": "gpt2", "stride": 512, "max_length": 1024, "reward_mode": "exp_neg_loss", "weight": 0.4}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

This configuration uses both **BERTScore** and **Perplexity** as components, with **BERTScore** weighted at **60%** and **Perplexity** weighted at **40%**. The reward is then **clipped** between `0` and `1` to prevent instability.

---

### 4. **Final Thoughts**

* **Modular Rewarding**: This setup allows you to easily switch between different reward functions and fine-tune their relative weights.
* **Hybrid Updates**: The `HybridTrainer` alternates between **Supervised Learning (SFT)** and **Reinforcement Learning (PPO)** in the same training loop, making it suitable for **fine-tuning and policy refinement**.
* **Reward Flexibility**: You can extend the reward system by adding more complex reward functions (e.g., ROUGE, BLEU, custom metrics).


---

This is an excellent addition to the **Hybrid Trainer**! You've effectively modularized and added **BLEU**, **ROUGE-L**, and **toxicity** rewards into the system. Now, you can extend the reward system by enabling these optional rewards based on whether the required dependencies are available. These rewards will be used seamlessly within your training pipeline.

Let’s walk through how you can integrate this into the **Hybrid Trainer** and the reward system:

---

### 1) **Integrating BLEU, ROUGE-L, and Toxicity Rewards into the `HybridTrainer`**

Here’s how you can incorporate these new reward classes into the **HybridTrainer**. I'll show how to:

* Add the reward classes to the reward factory.
* Call them in the `HybridTrainer` during the **PPO step** (where rewards are computed).

### **Reward Class Integration**

You’ve already created the classes for **BLEU**, **ROUGE-L**, and **Toxicity**, and the factory method for picking the appropriate reward is in place. So now, in your **HybridTrainer**, you just need to hook up these reward classes with your existing reward function setup.

---

### 2) **Reward Factory Integration**

Let's start by extending the reward factory to handle the new classes. Here's how the reward selection will work when you load your configuration.

#### **`_build_leaf` Update**:

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

### 3) **Reward Calculation in HybridTrainer**

Now that your factory is handling the reward classes, you need to integrate them into the **PPO** step within the `HybridTrainer`.

In your `HybridTrainer`, when you call `ppo_step()`, it will now compute rewards based on the configured rewards (`BLEU`, `ROUGE-L`, `Toxicity`, etc.) using the reward classes.

Here’s how the **PPO step** would look with your new rewards:

```python
class HybridTrainer:
    def __init__(self, model, tokenizer, train_data, reward_fn, 
                 lr_sft=5e-5, lr_ppo=1e-5, batch_size=4, maxseq=1024, 
                 max_new_tokens=128, sft_steps_per_cycle=1, ppo_steps_per_cycle=1):
        # Initialize basic components
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.reward_fn = reward_fn  # This is the reward factory you set up
        self.lr_sft = lr_sft
        self.lr_ppo = lr_ppo
        self.batch_size = batch_size
        self.maxseq = maxseq
        self.max_new_tokens = max_new_tokens
        self.sft_steps_per_cycle = sft_steps_per_cycle
        self.ppo_steps_per_cycle = ppo_steps_per_cycle

        # Initialize the optimizers for SFT and PPO
        self.opt_sft = AdamW(model.parameters(), lr=lr_sft)
        self.ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, learning_rate=lr_ppo)

    def ppo_step(self, batch):
        """
        Perform a Reinforcement Learning (PPO) step.
        """
        # Collect responses from the model based on the batch
        queries, responses = self.collect_responses(batch)

        # Calculate rewards using the reward_fn (which selects BLEU, ROUGE-L, Toxicity, etc.)
        rewards = [self.reward_fn(r) for r in responses]

        # Perform PPO update
        self.ppo_trainer.step(queries, responses, rewards)

        return sum(rewards) / len(rewards)
```

Now, your `ppo_step` will:

1. Generate model responses.
2. Compute the rewards using the **reward\_fn**, which will dynamically pick the appropriate reward class (e.g., **BLEU**, **ROUGE-L**, **Toxicity**, etc.).
3. Update the model using the PPO algorithm.

### 4) **JSON Configuration Example for Hybrid Reward System**

You can configure which reward systems to use in a flexible JSON format. Here’s how you can specify **BLEU**, **ROUGE-L**, and **Toxicity** rewards (mixing them with custom weights):

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "bleu", "smooth_method": "exp", "lowercase": true, "weight": 0.4},
      {"type": "rouge_l", "use_stemmer": true, "lowercase": true, "weight": 0.3},
      {"type": "toxicity", "model": "original", "invert": true, "weight": 0.3}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

This example configures a **weighted sum** of **BLEU**, **ROUGE-L**, and **Toxicity**, with each reward contributing to the total score with different weights. The `postprocess` clips the final reward to be in the range `[0, 1]`.

### 5) **Performance Considerations**

* **BLEU**: Requires tokenization and computing BLEU score (can be slow for long text). Use **smoothing** and adjust `tokenize` method for faster performance.
* **ROUGE-L**: ROUGE can be computation-heavy. If you're processing long texts, adjust the model and use the most efficient scoring method.
* **Toxicity**: The **Detoxify** model is light but still requires computing toxicity scores. You might want to batch it for efficiency.

---

### Final Integration Flow:

1. **Model & Tokenizer Setup**: Load the model and tokenizer as usual.
2. **Reward Configuration**: Use a JSON config to specify reward types (e.g., BLEU, ROUGE-L, Toxicity).
3. **Reward Calculation**: During training, the `HybridTrainer` will calculate rewards based on the configured rewards and perform PPO updates accordingly.
4. **Flexible Rewards**: Add more rewards in the future with minimal changes to your codebase.

---

### Next Steps:

* **Test it out** with a small dataset and experiment with different reward combinations.
* **Tune performance** based on your training setup and reward types (e.g., batch evaluation for **BLEU** or **ROUGE-L**).
* **Flashback to Adaptive Memory**: You could hook this into **Adaptive Memory** later to prioritize revisiting specific reward transitions based on **BLEU** or **ROUGE-L** scores.

Would you like to proceed with any of these next steps or focus on something else?

# Example dataset
train_data = [
    {"prompt": "How are you?", "output": "I'm doing great, thanks!"},
    {"prompt": "What's your name?", "output": "I am GPT."},
    {"prompt": "Where do you live?", "output": "I live in the cloud."}
]

# Example reward config for weighted rewards
reward_config = {
  "type": "weighted_sum",
  "components": [
      {"type": "bleu", "smooth_method": "exp", "lowercase": True, "weight": 0.4},
      {"type": "rouge_l", "use_stemmer": True, "lowercase": True, "weight": 0.3},
      {"type": "toxicity", "model": "original", "invert": True, "weight": 0.3}
  ],
  "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
  ]
}

# Define reward function using the configuration (this is a simplified example)
reward_fn = _build_leaf(reward_config)

# Initialize the HybridTrainer
model_name = "gpt2"  # You can change this to your own model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

trainer = HybridTrainer(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    reward_fn=reward_fn,
    lr_sft=5e-5, 
    lr_ppo=1e-5, 
    batch_size=4, 
    maxseq=1024, 
    max_new_tokens=128
)

# Start training
trainer.train(epochs=1)  # You can adjust the number of epochs as needed


---

I see what you're aiming for with this approach—an **extensible and modular reward factory** that seamlessly plugs into the **HybridTrainer**. This design allows you to dynamically select, combine, and process rewards in a way that supports both **Supervised Learning (SFT)** and **Reinforcement Learning (PPO)**. I really love how flexible it is, and it enables you to easily swap out rewards and adjust them without needing to change the core training logic.

### Let's break it down and integrate this into the **HybridTrainer** with a step-by-step approach.

---

### **Step 1: Plug the `rewards.py` Module into `HybridTrainer`**

You’ve already created the reward classes and factory methods, so now let’s ensure the **HybridTrainer** class can leverage them.

---

#### **1.1**: Importing the `rewards.py` Module

In your `HybridTrainer` script, add the import for the `make_reward` function:

```python
from rewards import make_reward
```

---

#### **1.2**: Update the `HybridTrainer` to use the Modular Reward System

Here’s the key part: **update the `HybridTrainer`** to use the reward factory to dynamically select and process rewards.

In your `HybridTrainer`, where the reward is calculated for PPO, use the reward factory:

```python
class HybridTrainer:
    def __init__(self, model, tokenizer, train_data, reward_spec, 
                 lr_sft=5e-5, lr_ppo=1e-5, batch_size=4, maxseq=1024, 
                 max_new_tokens=128, sft_steps_per_cycle=1, ppo_steps_per_cycle=1):
        # Initialize basic components
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.reward_fn = make_reward(reward_spec)  # Use the reward spec to build the reward function
        self.lr_sft = lr_sft
        self.lr_ppo = lr_ppo
        self.batch_size = batch_size
        self.maxseq = maxseq
        self.max_new_tokens = max_new_tokens
        self.sft_steps_per_cycle = sft_steps_per_cycle
        self.ppo_steps_per_cycle = ppo_steps_per_cycle

        # Initialize the optimizers for SFT and PPO
        self.opt_sft = AdamW(model.parameters(), lr=lr_sft)
        self.ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, learning_rate=lr_ppo)

    def ppo_step(self, batch):
        """
        Perform a Reinforcement Learning (PPO) step.
        """
        # Collect responses from the model based on the batch
        queries, responses = self.collect_responses(batch)

        # Calculate rewards using the reward_fn (which selects BLEU, ROUGE-L, Toxicity, etc.)
        rewards = [self.reward_fn(r) for r in responses]  # Dynamic reward calculation

        # Perform PPO update
        self.ppo_trainer.step(queries, responses, rewards)

        return sum(rewards) / len(rewards)
```

Now, **`reward_fn`** is dynamically built based on the configuration that is passed to the `HybridTrainer` when it’s instantiated. This allows you to configure the reward function via JSON.

---

### **Step 2: Using the Modular Reward System in the Trainer**

---

#### **2.1**: Sample JSON Configuration

Here’s an example of how you would configure a **weighted sum** of rewards like **Cosine Similarity**, **Length Window**, and **Exact Match**:

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "cosine_sts", "weight": 0.5},
      {"type": "length_window", "min": 30, "max": 100, "weight": 0.3},
      {"type": "exact_match", "weight": 0.2}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

You can load this JSON into your training script to pass the configuration to the **HybridTrainer**.

```python
# Load the JSON config for rewards
with open("reward_config.json", "r") as f:
    reward_spec = json.load(f)

# Initialize the HybridTrainer
trainer = HybridTrainer(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    reward_spec=reward_spec,  # Pass the reward config here
    lr_sft=5e-5, 
    lr_ppo=1e-5, 
    batch_size=4, 
    maxseq=1024, 
    max_new_tokens=128
)
```

---

#### **2.2**: Testing the Reward System

Once the **HybridTrainer** is set up with the **reward factory**, you can test the system on a small dataset. The rewards will be calculated dynamically based on the configuration and passed to PPO.

Here’s an example of how the rewards would be calculated during training:

```python
# Example training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    for step, batch in enumerate(train_data):
        # Step 1: Supervised Learning (SFT) step
        sft_loss = self.sft_step(batch)
        print(f"Step {step} | SFT loss: {sft_loss:.4f}")

        # Step 2: Reinforcement Learning (PPO) step
        response_text = model.generate(batch['prompt'])
        target_text = batch['output']

        # Calculate reward based on current response and target (using reward_fn)
        reward = self.reward_fn(response_text, target_text)
        print(f"Step {step} | Reward: {reward:.4f}")
        
        # PPO update
        self.ppo_step(batch)
```

---

### **Step 3: Post-Processing Rewards**

You can further **process** rewards using **clip**, **scale**, or **EMA normalize** as specified in the config. The post-processing happens at the **end of each reward calculation**:

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "cosine_sts", "weight": 0.7},
      {"type": "regex_presence", "pattern": "\\bthanks?\\b", "weight": 0.1},
      {"type": "length_window", "min": 40, "max": 160, "weight": 0.2}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1},   # Clipping rewards to [0, 1]
      {"type": "ema_normalize", "decay": 0.995} # EMA normalization for stability
    ]
  }
}
```

### **Step 4: Evaluation and Testing**

You can test the reward-driven **PPO updates** by:

* **Logging rewards**: This helps you monitor whether the model is improving based on the reward function.
* **Visualizing rewards**: Track reward trends and observe if the model shifts its behavior after certain updates.

---

### **Next Steps**:

* **Test on a small dataset** and make sure everything works with your configuration.
* **Tune reward weights** to balance different reward signals (e.g., semantics vs. length).
* **Implement Adaptive Memory**: Track which states result in the highest rewards and adaptively prioritize them.

---

### **Do you want to**:

1. **Run the training loop** with a test dataset to see how the rewards impact the model’s learning?
2. **Add custom reward types** or post-processing techniques for further customization?

Let me know how you'd like to proceed!

---

This is an excellent addition to the **HybridTrainer**! By incorporating **sentiment analysis** as a modular reward, you're enhancing the model's ability to understand the emotional tone of responses, which is a crucial aspect for many real-world NLP tasks, like customer support, conversational agents, and content moderation.

### Here's a breakdown of how to integrate sentiment analysis into your **HybridTrainer** system, following the pattern you've established.

---

### **Step 1: Add Sentiment Analysis Reward to `rewards.py`**

You can easily slot **sentiment analysis** as an additional reward leaf. As you’ve outlined, we’ll create a `SentimentReward` class that uses a Hugging Face pipeline for sentiment analysis.

#### **1.1: Add Sentiment Reward Class**

In `rewards.py`, add the following import and class definition for **SentimentReward**:

```python
# rewards.py (add near optional imports)
try:
    from transformers import pipeline
    _HAS_PIPELINE = True
except Exception:
    _HAS_PIPELINE = False

# -----------------------------
# Sentiment Analysis Reward
# -----------------------------
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

### **Step 2: Update Reward Factory**

In the `_build_leaf` function, add the **SentimentReward** to handle when the reward type is **sentiment**:

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
    if t == "sentiment":
        return SentimentReward(
            model=spec.get("model", "distilbert-base-uncased-finetuned-sst-2-english"),
            target=spec.get("target", "positive"),
            scale=bool(spec.get("scale", True)),
            device=spec.get("device"),
        )
    raise ValueError(f"Unknown reward type: {t}")
```

Now, **sentiment** can be dynamically selected as a reward type through the JSON configuration.

---

### **Step 3: Example JSON Configurations**

Now you can configure the **sentiment analysis** reward in your **JSON config** to guide the PPO model’s learning.

#### **3.1: Reward Positive Sentiment**

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

This configuration rewards **positive sentiment** responses by scaling the probability between `0` and `1`.

#### **3.2: Reward Negative Sentiment (Binary, Not Scaled)**

```json
{
  "REWARD": {
    "type": "sentiment",
    "target": "negative",
    "scale": false
  }
}
```

This configuration gives a binary reward: `1` for negative sentiment, `0` for non-negative sentiment.

#### **3.3: Blend Sentiment with Fluency (Perplexity)**

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

This combines **positive sentiment** and **fluency** (via **perplexity**) with equal weight, allowing the model to prioritize responses that are both **positive in tone** and **fluent**.

---

### **Step 4: Use Sentiment Reward in PPO Loop**

Once the reward is configured, it can be used in the **PPO loop**. After generating responses, you simply call the reward function:

```python
# after you get response_text and target_text
rew = reward_fn(response_text, target_text, prompt=prompt, item=item)
batch_rewards.append(float(rew))
```

This computes the sentiment reward for each response, which is then added to the **batch\_rewards**.

---

### **Step 5: Optional - Multi-Dimensional Sentiment**

If you want to include multiple sentiment dimensions (like **joy**, **anger**, or **sadness**) using something like `cardiffnlp/twitter-roberta-base-sentiment`, you can stack multiple sentiment analysis tasks in the same reward:

#### **Example: Multi-Dimensional Sentiment**

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "sentiment", "target": "joy", "scale": true, "weight": 0.3},
      {"type": "sentiment", "target": "anger", "scale": true, "weight": 0.3},
      {"type": "sentiment", "target": "sadness", "scale": true, "weight": 0.4}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

This would reward **joy**, **anger**, and **sadness** with weighted contributions, allowing the model to favor certain emotional tones over others, based on the configuration.

---

### **Final Thoughts**

* **Modular Rewards**: This allows you to extend your reward system further with any other custom **emotion**, **sentiment**, or **metric**.
* **Dynamic Reward Configuration**: With JSON config, you can adjust the reward components and their weights without modifying the core code.
* **Sentiment-Driven Learning**: By prioritizing positive or negative sentiment, you can steer your model’s behavior towards specific emotional tones, which is helpful for use cases like conversational agents, customer service bots, and content moderation.

---



This is an excellent way to integrate **truthfulness** as a reward into the **HybridTrainer**! Using a **Natural Language Inference (NLI)** model or **retrieval-augmented NLI** is a smart approach for assessing the validity or truthfulness of the model's responses.

Let's walk through the steps to seamlessly integrate this into the **HybridTrainer**.

---

### **Step 1: Add Truthfulness NLI and Retrieval-Augmented NLI to `rewards.py`**

First, add the required **Truthfulness NLI** and **Retrieval-Augmented Truthfulness** classes to `rewards.py`:

#### **1.1: Add Guarded Imports**

Add these imports at the top of your `rewards.py` to conditionally import necessary libraries:

```python
try:
    from transformers import pipeline as hf_pipeline
    _HAS_HF_PIPELINE = True
except Exception:
    _HAS_HF_PIPELINE = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util  # You already import this above
    _HAS_ST = True
except Exception:
    _HAS_ST = False
```

#### **1.2: Add the Truthfulness NLI Reward Class**

Add the following `TruthfulnessNLI` class to `rewards.py`:

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

        out = self.clf({"text": premise, "text_pair": hypothesis})[0]  # list of dicts with 'label' and 'score'

        # Map to probs
        scores = {d["label"].lower(): float(d["score"]) for d in out}
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

#### **1.3: Add the Retrieval-Augmented Truthfulness Reward Class**

Now, add the **TruthfulnessRAG** class, which uses **SentenceTransformers** for embedding-based retrieval and **NLI** for truthfulness validation.

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

---

### **Step 2: Add the Truthfulness Rewards to the Factory**

Update your `_build_leaf` function to support the new **TruthfulnessNLI** and **TruthfulnessRAG** rewards.

```python
def _build_leaf(spec: Dict[str, Any]) -> Reward:
    t = spec.get("type", "").lower()
    if t == "truth_nli":
        return TruthfulnessNLI(
            model=spec.get("model", "roberta-large-mnli"),
            device=spec.get("device"),
            aggregation=spec.get("aggregation", "p_e_minus_p_c"),
        )
    if t == "truth_rag":
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

### **Step 3: Example JSON Configurations**

#### **3.1: Truthfulness via NLI (Using a Reference)**

```json
{
  "REWARD": {
    "type": "truth_nli",
    "model": "roberta-large-mnli",
    "aggregation": "p_e_minus_p_c",
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

#### **3.2: Retrieval-Augmented Truthfulness**

```json
{
  "REWARD": {
    "type": "truth_rag",
    "embedder": "all-MiniLM-L6-v2",
    "nli_model": "roberta-large-mnli",
    "top_k": 5,
    "agg": "max",
    "corpus": [
      "The sky is blue.",
      "Water is wet.",
      "Einstein developed the theory of relativity."
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

---

### **Step 4: Integrate into the PPO Loop**

You can now plug the **truthfulness reward** directly into your PPO loop, as follows:

```python
# after you get response_text and target_text
truthfulness_reward = reward_fn(response_text, target_text, prompt=prompt, item=item)
batch_rewards.append(float(truthfulness_reward))
```

This calculates the truthfulness score for each response, and the score can be used in the PPO training loop.

---

### **Final Thoughts**

* **Truthfulness via NLI**: This allows the model to assess how **truthful** a response is, based on its agreement with a provided reference.
* **Retrieval-Augmented Truthfulness**: By adding a corpus of evidence, you can provide the model with **external information** to validate its responses, improving accuracy for open-domain queries.
* **Integration with HybridTrainer**: By combining these truthfulness rewards with other rewards like **perplexity**, **toxicity**, and **sentiment**, you can steer your model to generate more **truthful**, **coherent**, and **positive** responses


.

Let me know if you want me to assist with **testing**, **fine-tuning**, or **other reward types**!

---

# MHT.py

import torch
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW


class HybridTrainer:
    def __init__(self, model_name: str, train_data: torch.utils.data.Dataset, reward_fn: Optional[callable] = None,
                 optimizer: Optional[Optimizer] = None, scheduler: Optional[_LRScheduler] = None):
        """
        Initialize HybridTrainer with modular components.
        """
        self.model_name = model_name
        self.train_data = train_data  # dataset for supervised learning
        self.reward_fn = reward_fn    # callable reward function for reinforcement learning
        self.optimizer = optimizer if optimizer else AdamW(self.model.parameters(), lr=1e-5)  # default optimizer
        self.scheduler = scheduler    # Learning rate scheduler
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_model(self):
        """
        Load model from Hugging Face model hub or a local checkpoint.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.train()  # Set the model to training mode
        return model

    def train_supervised(self, epochs: int = 1, batch_size: int = 8):
        """
        Train the model in supervised learning mode (using labeled data).
        """
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=10_000,
            save_total_limit=2,
            eval_steps=500,  # Periodic evaluation
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            optimizers=(self.optimizer, self.scheduler)
        )

        trainer.train()

    def train_reinforcement(self, epochs: int = 1, batch_size: int = 8):
        """
        Train the model using reinforcement learning (with rewards).
        """
        for epoch in range(epochs):
            for step, batch in enumerate(self.train_data):
                inputs = self.tokenizer(batch['input'], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                targets = self.tokenizer(batch['output'], return_tensors="pt").to(self.model.device)
                
                # Forward pass (get model's predictions)
                outputs = self.model(**inputs, labels=targets['input_ids'])
                loss = outputs.loss

                # Compute reward from the reward function
                reward = self.reward_fn(outputs, batch)
                reward_loss = loss * reward  # Adjust loss by the reward

                # Backpropagate the loss
                self.optimizer.zero_grad()
                reward_loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                if step % 10 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss {reward_loss.item()}, Reward {reward}")

    def run_training(self, supervised_epochs: int = 1, rl_epochs: int = 1, batch_size: int = 8):
        """
        Run both supervised and reinforcement learning training.
        """
        print("Starting supervised training...")
        self.train_supervised(epochs=supervised_epochs, batch_size=batch_size)

        print("Starting reinforcement learning...")
        self.train_reinforcement(epochs=rl_epochs, batch_size=batch_size)

# MHT_train_loop.py

# MHT_train_loop.py
from datasets import load_dataset
from MHT import HybridTrainer
from MHT_Reward import SentimentReward

# Load the dataset (you can choose another dataset based on your task)
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

# Use the train split for training
train_data = dataset['train']

# If your dataset is not in the correct format (text), you might need to preprocess it:
def preprocess_function(examples):
    return {"input": examples["text"], "output": examples["text"]}

train_data = train_data.map(preprocess_function, remove_columns=["text"])

# Instantiate reward function (sentiment-based reward)
reward_fn = SentimentReward(target="positive")

# Create HybridTrainer instance with the dataset and reward function
trainer = HybridTrainer(
    model_name="gpt2",
    train_data=train_data,
    reward_fn=reward_fn
)

# Run training (first supervised, then reinforcement learning)
trainer.run_training(supervised_epochs=1, rl_epochs=3, batch_size=4)

The **HybridTrainer** is a powerful and modular framework that allows you to combine **supervised learning** (SL) with **reinforcement learning** (RL). You can customize this system with various reward functions, such as **sentiment analysis**, **truthfulness** (via NLI models), **fluency**, or any other metric relevant to your use case.

Below is the complete setup of how **HybridTrainer** integrates **sentiment analysis**, **truthfulness (NLI)**, and **perplexity** as rewards during the **RL training phase**, following the principles of modularity and adaptability.

### **1. HybridTrainer Setup**

We will create a `HybridTrainer` that uses both **Supervised Learning (SL)** and **Reinforcement Learning (RL)** in a combined training loop. It will alternate between these two paradigms based on the configuration.

### **Key Concepts in the HybridTrainer**:

* **Supervised Learning (SL)**: The model is initially trained on labeled data to understand basic patterns.
* **Reinforcement Learning (RL)**: The model is then fine-tuned with real-time feedback, optimizing for specific goals based on reward functions.
* **Modular Reward Functions**: The reward functions can be easily switched out or combined to guide the model to specific objectives, like sentiment analysis, truthfulness, or fluency.

---

### **2. Reward Functions Setup (Sentiment, Truthfulness, Perplexity)**

Let’s implement the reward functions, including **Sentiment**, **Truthfulness (NLI)**, and **Perplexity**. These will serve as the feedback mechanisms for the **RL** part of the training.

#### **2.1 Sentiment Reward**

We use a Hugging Face pipeline for sentiment analysis, providing a reward based on whether the model’s response has a **positive sentiment**.

```python
from transformers import pipeline

class SentimentReward:
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english", target="positive", scale=True):
        self.model = pipeline("sentiment-analysis", model=model)
        self.target = target
        self.scale = scale

    def __call__(self, response):
        result = self.model(response)[0]
        label = result["label"].lower()
        score = result["score"]
        if self.target in label:
            return score if self.scale else 1.0
        return 1.0 - score if self.scale else 0.0
```

#### **2.2 Truthfulness Reward using NLI**

This reward checks if the model’s response is consistent with a reference text, using a pre-trained **NLI model** (e.g., **roberta-large-mnli**). It computes the reward as $P(\text{entailment}) - P(\text{contradiction})$.

```python
from transformers import pipeline

class TruthfulnessReward:
    def __init__(self, model="roberta-large-mnli"):
        self.model = pipeline("text-classification", model=model, return_all_scores=True)

    def __call__(self, response, target):
        result = self.model({"text": target, "text_pair": response})[0]
        scores = {d["label"].lower(): float(d["score"]) for d in result}
        p_e = scores.get("entailment", 0.0)
        p_c = scores.get("contradiction", 0.0)
        return max(0.0, min(1.0, 0.5 * (p_e - p_c) + 0.5))
```

#### **2.3 Perplexity Reward**

We calculate **perplexity** using a pre-trained language model (e.g., **GPT2**), which helps measure how well the model's response fits within the distribution of a language model.

```python
import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class PerplexityReward:
    def __init__(self, model="gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.model.eval()  # Set to evaluation mode

    def __call__(self, response):
        inputs = self.tokenizer(response, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            return math.exp(loss.item())  # Perplexity is the exponential of the loss
```

---

### **3. HybridTrainer Class**

This class will manage both **supervised learning (SL)** and **reinforcement learning (RL)** stages. It will use the reward functions (e.g., **SentimentReward**, **TruthfulnessReward**, **PerplexityReward**) during the **RL** fine-tuning phase.

```python
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW

class HybridTrainer:
    def __init__(self, model, tokenizer, train_data, reward_fn, 
                 lr_sft=5e-5, lr_ppo=1e-5, batch_size=4, maxseq=1024, 
                 max_new_tokens=128, sft_steps_per_cycle=1, ppo_steps_per_cycle=1):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.reward_fn = reward_fn
        self.lr_sft = lr_sft
        self.lr_ppo = lr_ppo
        self.batch_size = batch_size
        self.maxseq = maxseq
        self.max_new_tokens = max_new_tokens
        self.sft_steps_per_cycle = sft_steps_per_cycle
        self.ppo_steps_per_cycle = ppo_steps_per_cycle

        # Initialize the optimizers for SFT and PPO
        self.opt_sft = AdamW(model.parameters(), lr=lr_sft)
        self.ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, learning_rate=lr_ppo)

    def sft_step(self, batch):
        """
        Perform a Supervised Learning (SFT) step.
        """
        inputs = self.tokenizer(batch['prompt'], return_tensors="pt", padding=True, truncation=True, max_length=self.maxseq)
        labels = self.tokenizer(batch['output'], return_tensors="pt")["input_ids"]
        
        # Forward pass (get model's predictions)
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss

        # Backpropagate the loss
        self.opt_sft.zero_grad()
        loss.backward()
        self.opt_sft.step()

        return loss.item()

    def ppo_step(self, batch):
        """
        Perform a Reinforcement Learning (PPO) step.
        """
        # Generate responses from the model based on the batch
        queries, responses = self.collect_responses(batch)

        # Calculate rewards using the reward_fn
        rewards = [self.reward_fn(r) for r in responses]

        # Perform PPO update
        self.ppo_trainer.step(queries, responses, rewards)

        return sum(rewards) / len(rewards)

    def collect_responses(self, batch):
        """
        Collect responses from the model for the given batch.
        """
        queries = []
        responses = []

        for item in batch:
            prompt = item['prompt']
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.maxseq)
            
            # Generate a response based on the prompt
            response_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

            queries.append(inputs['input_ids'])
            responses.append(response)

        return queries, responses

    def train(self, epochs=1):
        """
        Train the model alternating between SL and RL steps.
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Shuffle training data each epoch
            train_data_shuffled = self.train_data.sample(frac=1).reset_index(drop=True)

            # Training loop
            for step, batch in enumerate(train_data_shuffled.iterrows()):
                batch = batch[1]  # Unwrap to get the actual row of data
                
                # Step 1: Supervised Learning step
                sft_loss = self.sft_step(batch)
                print(f"Step {step} | SFT loss: {sft_loss:.4f}")

                # Step 2: Reinforcement Learning step
                ppo_reward = self.ppo_step(batch)
                print(f"Step {step} | PPO reward: {ppo_reward:.4f}")

                # Save model after each cycle or periodically
                if step % 100 == 0:
                    self.model.save_pretrained("path_to_save_model")
                    self.tokenizer.save_pretrained("path_to_save_model")
                    print(f"Model saved at step {step}")

```

---

### **4. JSON Configuration Example for Hybrid Reward System**

Now, you can configure which reward systems to use in a flexible JSON format. Here’s how you can specify **Sentiment**, **Truthfulness**, and **Perplexity** rewards (mixing them with custom weights):

#### **Example JSON Configuration**

```json
{
  "REWARD": {
    "type": "weighted_sum",
    "components": [
      {"type": "sentiment", "target": "positive", "scale": true, "weight": 0.4},
      {"type": "truth_nli", "aggregation": "p_e_minus_p_c", "weight": 0.3},
      {"type": "perplexity", "model": "gpt2-medium", "weight": 0.3}
    ],
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

This configuration combines **Sentiment**, **Truthfulness**, and **Perplexity** rewards with different weights, and clips the final reward between 0 and 1.

---

### **5. Running Training**

After setting up the **HybridTrainer** and reward system, you can begin training:

```python
trainer = HybridTrainer(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    reward_fn=reward_fn
)

trainer.train(epochs=3)  # Train for 3 epochs
```

---

### **Final Thoughts**

* **Modular Rewarding**: This design allows you to flexibly add or remove reward functions based on the task at hand.
* **Supervised and Reinforcement Learning**: The model alternates between **SL** (for foundational training) and **RL** (for fine-tuning based on custom rewards).
* **Extensible**: You can easily add new reward functions or adjust their weights using the **JSON config**.

This setup is great for applications like **conversational AI**, **content moderation**, and **knowledge generation**, where you want to guide the model to produce **truthful**, **positive**, and **coherent** outputs.

Let me know if you have any questions or would like to extend this further!

---

This is a solid approach to integrating **truthfulness** into your **HybridTrainer** system! By using a **Natural Language Inference (NLI)** model, you're able to assess the **truthfulness** of the model's responses dynamically. Here's how we can integrate this into the **HybridTrainer** using **truthfulness** rewards.

### **Step 1: Add Truthfulness Reward to `rewards.py`**

First, we'll define the **TruthfulnessReward** class, which uses an **NLI model** (such as `roberta-large-mnli`) to assess how truthful the model's responses are in relation to a provided reference (target).

#### **1.1: Add the Truthfulness NLI Reward Class**

Here's a class that uses NLI to determine whether the model’s response is truthful by comparing it to a reference text (i.e., `target`).

```python
from transformers import pipeline

class TruthfulnessReward:
    def __init__(self, model="roberta-large-mnli", device="cpu"):
        """
        Initialize the Truthfulness reward by loading an NLI model from Hugging Face.
        """
        self.model = pipeline("text-classification", model=model, return_all_scores=True, device=0 if device == "cuda" else -1)

    def __call__(self, response, target):
        """
        Calculate the truthfulness score using NLI. The reward is based on the entailment score.
        """
        result = self.model({"text": target, "text_pair": response})[0]
        scores = {d["label"].lower(): float(d["score"]) for d in result}
        p_e = scores.get("entailment", 0.0)
        p_c = scores.get("contradiction", 0.0)

        # Normalize the score between 0 and 1: reward = (P(entailment) - P(contradiction)) / 2 + 0.5
        return max(0.0, min(1.0, 0.5 * (p_e - p_c) + 0.5))
```

### **Step 2: Modify the Reward Factory**

Now, we need to add the **TruthfulnessReward** class to the reward factory, so it can be selected dynamically based on the configuration.

```python
def _build_leaf(spec: Dict[str, Any]) -> Reward:
    t = spec.get("type", "").lower()
    
    # Check for the truthfulness reward in the configuration
    if t == "truthfulness":
        return TruthfulnessReward(model=spec.get("model", "roberta-large-mnli"))
    
    # Other reward types can be added here...
    # Example: "sentiment", "perplexity", etc.
    
    raise ValueError(f"Unknown reward type: {t}")
```

### **Step 3: JSON Configuration for Truthfulness Reward**

You can configure the **TruthfulnessReward** in a flexible JSON format to specify the NLI model and other parameters. Here's an example:

#### **Example JSON Configuration**

```json
{
  "REWARD": {
    "type": "truthfulness",
    "model": "roberta-large-mnli",
    "postprocess": [
      {"type": "clip", "min": 0, "max": 1}
    ]
  }
}
```

This configuration uses the **`roberta-large-mnli`** model for **truthfulness assessment**. The reward score is **clipped** between `0` and `1` for stability.

### **Step 4: Integrating the Reward in the PPO Loop**

Once the **TruthfulnessReward** is configured, you can use it within the **PPO step** of your **HybridTrainer**. Here's how to apply the reward:

```python
class HybridTrainer:
    def __init__(self, model, tokenizer, train_data, reward_fn, 
                 lr_sft=5e-5, lr_ppo=1e-5, batch_size=4, maxseq=1024, 
                 max_new_tokens=128, sft_steps_per_cycle=1, ppo_steps_per_cycle=1):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.reward_fn = reward_fn  # The truthfulness reward function will be passed here
        self.lr_sft = lr_sft
        self.lr_ppo = lr_ppo
        self.batch_size = batch_size
        self.maxseq = maxseq
        self.max_new_tokens = max_new_tokens
        self.sft_steps_per_cycle = sft_steps_per_cycle
        self.ppo_steps_per_cycle = ppo_steps_per_cycle

        # Initialize the optimizers for SFT and PPO
        self.opt_sft = AdamW(model.parameters(), lr=lr_sft)
        self.ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, learning_rate=lr_ppo)

    def ppo_step(self, batch):
        """
        Perform a Reinforcement Learning (PPO) step.
        """
        # Generate responses from the model based on the batch
        queries, responses = self.collect_responses(batch)

        # Calculate rewards using the reward_fn (TruthfulnessReward)
        rewards = [self.reward_fn(r, batch['output']) for r in responses]

        # Perform PPO update
        self.ppo_trainer.step(queries, responses, rewards)

        return sum(rewards) / len(rewards)
```

### **Step 5: Example Usage in Training**

Finally, you can use the **HybridTrainer** to train your model with **truthfulness** as a reward in combination with other rewards.

```python
# Load the dataset
from datasets import load_dataset
train_data = load_dataset("wikitext", "wikitext-103-raw-v1")["train"]

# Define the reward function
reward_fn = TruthfulnessReward(model="roberta-large-mnli")

# Initialize the HybridTrainer with the truthfulness reward
trainer = HybridTrainer(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    reward_fn=reward_fn
)

# Run training
trainer.train(epochs=3)
```

### **Final Thoughts**

* **Truthfulness as a Reward**: By using NLI models, you can effectively guide the model to generate **truthful** responses based on a reference.
* **Modular Design**: The modular reward framework allows you to easily swap out or add additional rewards (e.g., **sentiment**, **perplexity**) based on your task requirements.
* **Flexible Training**: The integration with **PPO** ensures that the model can be fine-tuned dynamically with **real-time feedback** from the reward function.

This approach will allow you to create a **more reliable model** that is not only fluent and relevant but also grounded in **truthful information**, which is crucial for many real-world applications like conversational agents and knowledge generation.

Let me know if you'd like to expand on any part of this or need further assistance!

---

MHT.py 
