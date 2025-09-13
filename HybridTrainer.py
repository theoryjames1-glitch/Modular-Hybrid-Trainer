import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from torch.optim import AdamW

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

        # Calculate rewards
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

# Initialize model, tokenizer, and reward function
model_name = "gpt2"  # Change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# Example reward function (use your custom reward function)
def reward_fn(response):
    return 1.0  # dummy function, you should use your actual reward model

# Load training data (replace with your actual data loading mechanism)
train_data = [
    {"prompt": "How are you?", "output": "I'm good, thanks!"},
    {"prompt": "What's your name?", "output": "My name is GPT."}
]  # Example dummy data, replace with your actual data

# Initialize the trainer
trainer = HybridTrainer(model, tokenizer, train_data, reward_fn)

# Start training
trainer.train(epochs=1)
