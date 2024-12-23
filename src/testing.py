import torchmetrics
import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self):
        self.metrics = {
            'accuracy': torchmetrics.Accuracy(),
            'precision': torchmetrics.Precision(),
            'recall': torchmetrics.Recall(),
            'f1': torchmetrics.F1Score()
        }
        
    def evaluate_model(self, model, test_datasets: Dict):
        """Evaluate model performance across different tasks."""
        try:
            results = {}
            
            for task_name, dataset in test_datasets.items():
                task_metrics = self._evaluate_task(model, dataset, task_name)
                results[task_name] = task_metrics
                
            logger.info("Successfully evaluated model performance")
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {str(e)}")
            raise
            
    def _evaluate_task(self, model, dataset, task_name):
        """Evaluate model on specific task."""
        model.eval()
        metrics = {}
        
        with torch.no_grad():
            predictions = model(dataset.features)
            
            for metric_name, metric in self.metrics.items():
                score = metric(predictions, dataset.labels)
                metrics[metric_name] = score.item()
                
        return metrics
