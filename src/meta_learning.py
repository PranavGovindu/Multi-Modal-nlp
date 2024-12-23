import torch
import torch.nn as nn
from torch.optim import Adam
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MetaLearner:
    def __init__(self, model, learning_rate=0.001, meta_learning_rate=0.01):
        self.model = model
        self.meta_optimizer = Adam(self.model.parameters(), lr=meta_learning_rate)
        self.task_learning_rate = learning_rate
        
    def meta_learning_task_adaptation(self, task_data: Dict[str, List]):
        """Adapt model to new tasks using meta-learning."""
        try:
            # Initialize task-specific parameters
            task_parameters = []
            meta_loss = 0.0
            
            for task_name, task_dataset in task_data.items():
                # Split into support and query sets
                support_set, query_set = self._split_task_data(task_dataset)
                
                # Create task-specific model copy
                task_model = self._clone_model()
                task_optimizer = Adam(task_model.parameters(), lr=self.task_learning_rate)
                
                # Inner loop: adapt to specific task
                task_loss = self._adapt_to_task(
                    task_model,
                    task_optimizer,
                    support_set
                )
                
                # Evaluate on query set
                query_loss = self._evaluate_task(task_model, query_set)
                meta_loss += query_loss
                
                task_parameters.append(task_model.state_dict())
                
            # Outer loop: meta-update
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            logger.info("Successfully performed meta-learning adaptation")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to perform meta-learning: {str(e)}")
            raise
            
    def _split_task_data(self, task_dataset, support_size=0.8):
        """Split task data into support and query sets."""
        split_idx = int(len(task_dataset) * support_size)
        return task_dataset[:split_idx], task_dataset[split_idx:]
        
    def _clone_model(self):
        """Create a copy of the model for task-specific adaptation."""
        clone = type(self.model)()
        clone.load_state_dict(self.model.state_dict())
        return clone
