from torch.utils.data import DataLoader
import torch
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class FeedbackTuner:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def user_feedback_for_tuning(self, user_feedback: List[Dict], validation_data=None):
        """Fine-tune model based on user feedback."""
        try:
            # Prepare feedback data
            feedback_dataset = self._prepare_feedback_data(user_feedback)
            feedback_loader = DataLoader(feedback_dataset, batch_size=32, shuffle=True)
            
            # Fine-tune model
            for epoch in range(5):  # Limited epochs for quick adaptation
                epoch_loss = self._train_epoch(feedback_loader)
                
                # Validate if validation data is provided
                if validation_data:
                    val_loss = self._validate(validation_data)
                    logger.info(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
                
            logger.info("Successfully fine-tuned model with user feedback")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to fine-tune model: {str(e)}")
            raise
            
    def _prepare_feedback_data(self, feedback):
        """Convert user feedback to training data."""
        inputs = [f['input'] for f in feedback]
        corrections = [f['correction'] for f in feedback]
        return list(zip(inputs, corrections))
        
    def _train_epoch(self, dataloader):
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
