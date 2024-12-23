from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import uncertainty_scores
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ActiveLearningSystem:
    def __init__(self, base_model=None):
        self.model = base_model or RandomForestClassifier()
        self.uncertainty_threshold = 0.7
        
    def active_learning(self, user_feedback, new_data, labels=None):
        """Update model using active learning and user feedback."""
        try:
            # Process user feedback
            if user_feedback:
                self._incorporate_feedback(user_feedback)
            
            # Handle new data
            if new_data is not None:
                selected_indices = self._select_informative_samples(new_data)
                
                if labels is not None:
                    self._update_model(
                        new_data[selected_indices],
                        labels[selected_indices]
                    )
                
            logger.info("Successfully updated model with new data")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to perform active learning: {str(e)}")
            raise
            
    def _select_informative_samples(self, data):
        """Select most informative samples based on uncertainty."""
        predictions = self.model.predict_proba(data)
        uncertainties = uncertainty_scores(predictions)
        return np.where(uncertainties > self.uncertainty_threshold)[0]
        
    def _incorporate_feedback(self, feedback):
        """Incorporate user feedback into the model."""
        X_feedback = np.array([f['data'] for f in feedback])
        y_feedback = np.array([f['label'] for f in feedback])
        self._update_model(X_feedback, y_feedback)
        
    def _update_model(self, X, y):
        """Update the model with new training data."""
        self.model.partial_fit(X, y)
