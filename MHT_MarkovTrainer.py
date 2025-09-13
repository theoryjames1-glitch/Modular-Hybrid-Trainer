class HybridTrainer:
    def __init__(self, model, tokenizer, train_data, reward_fn, markov_memory: MarkovMemory, 
                 lr_sft=5e-5, lr_ppo=1e-5, batch_size=4, maxseq=1024, 
                 max_new_tokens=128, sft_steps_per_cycle=1, ppo_steps_per_cycle=1):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.reward_fn = reward_fn
        self.markov_memory = markov_memory  # Markov Memory is passed as an argument
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

        # Add experience to Markov Memory
        for prompt, response, reward in zip(batch['prompt'], responses, rewards):
            self.markov_memory.add_to_memory(prompt, response, reward)
        
        # Update the transition matrix after each batch
        self.markov_memory.update_transition_matrix()

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
