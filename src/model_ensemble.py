from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DynamicEnsemble:
    def __init__(self, models: List[Dict]):
        self.models = models
        self.task_performances = self._initialize_performances()
        
    def ensemble_models(self, input_data, task_type):
        """Combine multiple models dynamically based on task."""
        try:
            # Select appropriate models for the task
            selected_models = self._select_models(task_type)
            
            # Get predictions from all selected models
            predictions = []
            weights = []
            
            for model in selected_models:
                pred = model['model'].predict(input_data)
                predictions.append(pred)
                weights.append(model['performance'][task_type])
                
            # Combine predictions using weighted average
            weights = np.array(weights) / sum(weights)
            ensemble_prediction = self._combine_predictions(predictions, weights)
            
            logger.info(f"Successfully generated ensemble prediction for {task_type}")
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Failed to generate ensemble prediction: {str(e)}")
            raise
            
    def _select_models(self, task_type, top_k=3):
        """Select top performing models for the given task."""
        return sorted(
            self.models,
            key=lambda x: x['performance'][task_type],
            reverse=True
        )[:top_k]
        
    def _combine_predictions(self, predictions, weights):
        """Combine predictions using weighted average."""
        return np.average(predictions, weights=weights, axis=0)
        
    def _initialize_performances(self):
        """Initialize performance metrics for each model."""
        return {
            model['name']: {
                'summarization': model.get('summarization_score', 0.5),
                'qa': model.get('qa_score', 0.5),
                'classification': model.get('classification_score', 0.5)
            }
            for model in self.models
        }
