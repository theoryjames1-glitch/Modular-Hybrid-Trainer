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
