import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

class BiasDetector:
    def __init__(self, sensitive_attributes):
        self.sensitive_attributes = sensitive_attributes
        self.debiasing_strength = 0.1
        
    def detect_and_mitigate_bias(self, model, dataset):
        """Detect and mitigate bias in model outputs."""
        try:
            # Detect bias
            bias_metrics = self._measure_bias(model, dataset)
            
            if self._requires_debiasing(bias_metrics):
                # Apply adversarial debiasing
                debiased_model = self._apply_adversarial_debiasing(
                    model,
                    dataset,
                    bias_metrics
                )
                
                # Validate debiasing
                new_bias_metrics = self._measure_bias(debiased_model, dataset)
                
                logger.info("Successfully applied bias mitigation")
                return debiased_model, new_bias_metrics
            
            return model, bias_metrics
            
        except Exception as e:
            logger.error(f"Failed to detect/mitigate bias: {str(e)}")
            raise
            
    def _measure_bias(self, model, dataset):
        """Measure bias across sensitive attributes."""
        bias_metrics = {}
        
        for attribute in self.sensitive_attributes:
            predictions = model(dataset.features)
            disparate_impact = self._calculate_disparate_impact(
                predictions,
                dataset.labels,
                dataset[attribute]
            )
            bias_metrics[attribute] = disparate_impact
            
        return bias_metrics
        
    def _calculate_disparate_impact(self, predictions, labels, sensitive_attr):
        """Calculate disparate impact ratio."""
        positive_rate_0 = np.mean(predictions[sensitive_attr == 0])
        positive_rate_1 = np.mean(predictions[sensitive_attr == 1])
        return min(positive_rate_0 / positive_rate_1, positive_rate_1 / positive_rate_0)
