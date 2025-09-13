# MHT_MarkovMemory.py
"""### **Markov Chain Memory (MCM)**

Markov Chain Memory is a concept derived from the **Markov Process**, where the future state of a system depends only on the current state and not on the sequence of events that preceded it. This makes it a **memoryless** process (in the sense that the next state depends on the present state, and not on how the system arrived at the present state). When applied to systems like **language models** or **reinforcement learning**, the "memory" aspect refers to the system’s ability to store and recall certain states (or tokens) and their probabilities over time.

Let’s break down how it works in various contexts, especially in the context of **language models** and **reinforcement learning**:

---

### **1. Markov Chain Basics**

A **Markov Chain** is a sequence of random variables where:

* **States**: The possible states (or observations) the system can be in.
* **Transition Probabilities**: The probabilities of moving from one state to another.
* **Memorylessness**: The key property is that the system's next state depends only on the current state, not the history of states that led to the present state. This is known as the **Markov property**.

#### Example:

In a simple **weather model**:

* States: {Sunny, Cloudy, Rainy}
* Transition probabilities:

  * P(Sunny -> Cloudy) = 0.3
  * P(Cloudy -> Rainy) = 0.5
  * P(Rainy -> Sunny) = 0.4

Here, the weather tomorrow only depends on today's weather, and not how we got to today.

---

### **2. Markov Chain in Language Models**

In **language models**, a Markov Chain is often used to model the sequence of words (or tokens) in a sentence, where each word depends only on the previous word (or a small context window). The **Markov Chain Memory** can be extended to handle multiple words and even long-term dependencies using techniques like **n-grams** or **Recurrent Neural Networks (RNNs)**.

#### **Markov Chain Memory in Text Generation**:

A language model with **Markov Chain Memory** will predict the next word based only on the current word (or state), using transition probabilities learned during training. For example, in an **n-gram model**, the model looks at the last **n-1** words to predict the next word.

##### **N-gram Example**:

* A **2-gram** (bigram) model:

  * States: A sequence of 2 words.
  * Transition probabilities: The probability of the next word given the previous word.
  * For example, P("sunny" | "today") might be higher than P("sunny" | "cloudy").

While **Markov Chains** have memory (in the sense that they remember the most recent state), they typically have limited **context**. In **higher-order Markov models**, you consider longer sequences of words (n-grams).

##### **Markov Chain vs. Deep Models**:

* **Markov Chain** models (like n-grams) are simpler and work well for short-term dependencies.
* **Deep Learning Models** (like **LSTMs** and **Transformers**) can model **long-range dependencies** and more complex patterns than traditional Markov models.

---

### **3. Markov Chain Memory in Reinforcement Learning**

In **Reinforcement Learning (RL)**, **Markov Chain Memory** is important in the context of **Markov Decision Processes (MDPs)**, where an agent makes decisions based on the current state it’s in and attempts to maximize rewards. The memory aspect refers to how an agent maintains a representation of its environment.

#### **Markov Decision Process (MDP)**:

An MDP is defined by:

* **States**: The set of possible states the environment can be in.
* **Actions**: The set of actions the agent can take.
* **Transition probabilities**: The probability of transitioning from one state to another after taking an action.
* **Reward function**: The reward the agent gets for being in a particular state and taking a particular action.

In RL, the **Markov Property** implies that the current state fully captures all necessary information about the environment, and the agent’s decision-making depends only on the current state.

##### **Q-Learning** Example:

The agent chooses actions based on a **Q-table** that stores the expected future rewards for each state-action pair. The table is updated based on **Bellman equations** using the current state and action.

---

### **4. Markov Chain Memory in Language Models (Example)**

Let’s take a simple example where we use **Markov Chain Memory** in a language model to generate text. Here’s how it might work:

#### **Step-by-Step Process**:

1. **Training**:

   * Train the model on a large corpus of text.
   * For each token, calculate the transition probabilities of one word leading to another (based on observed frequencies).
   * Store these probabilities in a **transition matrix** (for n-grams or word pairs).

2. **Generation**:

   * Start with a seed word.
   * Use the transition matrix to sample the next word based on the current word.
   * Repeat the process, generating a sequence of words.

#### **Markov Chain Memory Example in Text Generation**:

For a sentence:

* **State space**: All possible words (or tokens).
* **Transition probabilities**: These are learned based on how likely a word is to follow another word (from training data).

  * P("sunny" | "weather") = 0.8
  * P("rainy" | "weather") = 0.2
  * P("cloudy" | "weather") = 0.4

---

### **5. Limitations of Markov Chain Memory**

* **Limited Context**: Standard Markov Chains usually consider only the **current state** and ignore the past. This leads to the **memorylessness** problem, where long-term dependencies can't be captured well.
* **Scalability**: For large datasets or long-term sequences (like text), storing all possible transitions in memory can become inefficient.
* **Lack of Complex Patterns**: Markov models can struggle with modeling complex, non-linear relationships in data (like **long-term dependencies** in language).

### **6. Enhancements Beyond Markov Chain Memory**

To address the limitations, enhancements like **n-grams**, **Hidden Markov Models (HMMs)**, **RNNs**, and **Transformers** are often used:

* **Hidden Markov Models**: Introduces hidden states to model more complex dependencies.
* **RNNs/LSTMs/Transformers**: These models maintain more complex and longer-term memory, which can capture dependencies over long sequences (unlike traditional Markov Chains).

---

### **Markov Chain Memory Use Cases**

1. **Text Generation**: Using n-gram models or more advanced models like HMMs for probabilistic text generation.
2. **Speech Recognition**: Markov Chains can model sequences of phonemes or words in speech recognition systems.
3. **Reinforcement Learning**: In RL, Markov Decision Processes (MDPs) rely on the Markov property to make decisions based on current states.
4. **Game AI**: AI systems in games use Markov Chains to model states and transitions, making decisions based on the current state of the game.

---

### **Summary**

* **Markov Chain Memory** is based on the idea that the future depends only on the present state, not on how the system arrived there.
* It works well for tasks with **short-term dependencies** (e.g., n-grams in language models).
* **Markov Chains** can be extended to more complex systems like **Hidden Markov Models (HMMs)** or **RNNs** for better long-term memory and sequence modeling.
* In **reinforcement learning**, the **Markov Decision Process** (MDP) is an essential concept where decisions are made based on the current state to maximize rewards."""
import numpy as np
from collections import defaultdict
import random

class MarkovMemory:
    def __init__(self, memory_size=1000, state_size=1024, alpha=0.1, gamma=0.9):
        """
        Markov Memory for tracking state transitions.

        Args:
            memory_size (int): Maximum number of states in memory.
            state_size (int): Size of each state (prompt-response pair).
            alpha (float): Learning rate for updating transitions.
            gamma (float): Discount factor for future rewards.
        """
        self.memory_size = memory_size
        self.state_size = state_size
        self.alpha = alpha
        self.gamma = gamma
        
        # Transition matrix to store state transitions and probabilities
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.state_rewards = defaultdict(float)  # Reward for each state
        
        # History and memory tracking
        self.memory = []
        self.state_history = []

    def _get_state_key(self, prompt, response):
        """
        Convert the prompt-response pair into a unique state key.
        """
        return (prompt, response)

    def add_to_memory(self, prompt, response, reward):
        """
        Add experience (prompt-response pair, reward) to the memory.
        """
        state = self._get_state_key(prompt, response)
        self.memory.append((state, reward))
        self.state_rewards[state] = reward

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # Remove the oldest memory entry

    def update_transition_matrix(self):
        """
        Update the transition matrix based on the current state and reward.
        """
        for i in range(1, len(self.memory)):
            prev_state, prev_reward = self.memory[i - 1]
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

Yes, you can absolutely use **Markov Memory** during inference. The idea of **Markov Memory** is based on storing past states (prompts, responses, actions) in a memory structure (usually a **transition matrix** or a **Markov Chain**) and using this memory to influence the next action or response based on historical data.

### How Markov Memory Works During Inference:

1. **Transition Matrix**: During training, you will build a **transition matrix** that stores the probability of moving from one state to another. These states could represent **sequences of tokens** or **states in dialogue**. When inference is performed, the system will **look up** the previous state and select the most likely next state (response) based on the transition probabilities stored in the memory.

2. **State Representation**: During inference, you can track the **current state** (such as the input prompt or a sequence of tokens) and use the transition matrix to determine the **probability of subsequent states** (next words, sentences, or responses). This allows you to generate more **contextually coherent** and **consistent** responses over time.

3. **Memory Utilization**: The **Markov Chain** or **transition matrix** is updated during inference based on **new interactions**. For example, if a certain prompt and response pair leads to a successful interaction (e.g., high reward or user satisfaction), the system can prioritize revisiting that pair during subsequent steps.

### Example of Markov Memory Usage in Inference:

Let's break it down with a simple example of using **Markov Memory** during inference:

### 1. **Build the Transition Matrix**: During the training phase, the model records transitions between **prompt-response pairs**.

```python
class MarkovChainMemory:
    def __init__(self):
        self.transition_matrix = {}
        self.state_rewards = {}
    
    def update(self, state, next_state, reward):
        if state not in self.transition_matrix:
            self.transition_matrix[state] = {}
        if next_state not in self.transition_matrix[state]:
            self.transition_matrix[state][next_state] = 0
        self.transition_matrix[state][next_state] += reward
    
    def get_next_state(self, current_state):
        if current_state in self.transition_matrix:
            next_states = self.transition_matrix[current_state]
            return max(next_states, key=next_states.get)  # Select the most probable next state
        return None
```

### 2. **Using Markov Memory During Inference**: During inference, you will use the **Markov Chain Memory** to select the next most likely state (response) based on the historical states.

```python
class MarkovMemoryInference:
    def __init__(self, markov_memory, gpt_model, tokenizer):
        self.markov_memory = markov_memory  # Markov Chain Memory
        self.gpt_model = gpt_model
        self.tokenizer = tokenizer
    
    def generate_response(self, prompt, max_new_tokens=50):
        """
        Generate a response using Markov Memory and GPT.
        """
        # Check if there's a prior state
        previous_state = self.get_previous_state(prompt)
        
        if previous_state:
            # Use Markov Memory to predict next state
            predicted_state = self.markov_memory.get_next_state(previous_state)
        else:
            predicted_state = None
        
        # If we have a predicted state, use it to help generate the response
        if predicted_state:
            prompt = predicted_state  # Update prompt with the Markov memory
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate a response using GPT (could also pass the context/state memory here)
        response_ids = self.gpt_model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Update the Markov memory with the new response
        self.markov_memory.update(prompt, response, reward=1)  # Assuming the reward is positive for now
        
        return response
    
    def get_previous_state(self, prompt):
        """
        Retrieve the previous state or memory if available.
        """
        return prompt  # In this case, the prompt itself is the state. Adjust as needed.
```

### How It Works:

1. **Markov Chain Memory**:

   * During training, each **state transition** is recorded with a reward. A **state** could be a specific prompt, and a **next\_state** would be the expected response.
   * The **transition matrix** stores how likely one state leads to another, with a **reward** associated with the transition.

2. **Inference**:

   * During inference, the **Markov Chain Memory** is consulted to check if the current state (prompt) has a **previous transition** stored.
   * If a previous state exists in memory, the model uses it to **select the next most likely response** based on the transition matrix.
   * The **GPT model** is then used to generate a response based on the current state and, optionally, any memory context (e.g., hidden state from the **RNN** or the **Markov Memory**).

3. **Updating Memory**:

   * After generating the response, the **transition matrix** is updated with the **new response** and associated **reward**.
   * This helps the model **adapt and refine its memory** based on real-time interactions.

### Example of Using Markov Memory in a Dialogue System:

For example, let's say the system has learned that the response to "How are you?" should be something like "I'm good, thank you!" based on past conversations. During inference, when the prompt "How are you?" is given again, the system can refer to its **Markov memory** to recall that the most probable response to that prompt is "I'm good, thank you!" (if it was a successful transition previously).

### **Benefits of Using Markov Memory in Inference:**

1. **Context-Aware**: By leveraging memory (Markov Chain), the system can maintain **context** over time, making it more suitable for tasks like **dialogue systems** or **story generation**.

2. **Efficiency**: The **Markov Chain** allows for quick lookup of the most probable next state, avoiding the need to store a full history of interactions while still maintaining context.

3. **Adaptability**: The memory can evolve as the model encounters more interactions. The system can **learn new patterns** and adjust its responses based on previous states that were successful.

4. **Improved Coherence**: Memory-based systems ensure that the responses are not randomly generated but are instead **consistent** with past interactions, which is crucial for tasks requiring **coherence** over time (e.g., dialogue systems).

### **Limitations to Consider**:

* **Memory Size**: Depending on how much historical data you want to retain, the **transition matrix** could grow large. You may need to **prune** the memory or limit the size of the context.
* **Transition Probabilities**: Markov models typically assume that the next state depends only on the current state (not on past states). For more complex interactions, this could limit the model’s ability to handle **longer-term dependencies**.

---

### **Conclusion**:

Yes, you can use **Markov Memory during inference**! By leveraging the **transition matrix** built during training, you can maintain context, make decisions based on past interactions, and generate responses that are **more coherent** and **consistent** with previous states. This approach can significantly enhance applications like **dialogue systems**, **personal assistants**, or **story generation** where maintaining long-term context is crucial.

Would you like help integrating this with your existing models, or would you like to explore other memory types (like **LSTMs** or **RNNs**) for long-term dependencies?
