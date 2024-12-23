from typing import Dict, List
import openai
import logging

logger = logging.getLogger(__name__)

class PromptOptimizer:
    def __init__(self):
        self.templates = {
            "summarization": "Summarize the following text:\n{input}\n\nKey points:",
            "qa": "Context: {input}\n\nQuestion: {question}\nAnswer:",
            "analysis": "Analyze the following text for key insights:\n{input}\n\nInsights:"
        }
        
    def optimize_prompt_for_task(self, task: str, input_data: str, 
                               additional_context: Dict = None) -> str:
        """Generate optimized prompts based on task and input data."""
        try:
            # Select base template
            template = self._select_template(task)
            
            # Analyze input characteristics
            input_features = self._analyze_input(input_data)
            
            # Optimize template based on input features
            optimized_prompt = self._customize_template(
                template, 
                input_features, 
                additional_context
            )
            
            logger.info(f"Successfully optimized prompt for task: {task}")
            return optimized_prompt
            
        except Exception as e:
            logger.error(f"Failed to optimize prompt: {str(e)}")
            raise
            
    def _select_template(self, task: str) -> str:
        """Select appropriate template based on task type."""
        return self.templates.get(task.lower(), self.templates["analysis"])
        
    def _analyze_input(self, input_data: str) -> Dict:
        """Analyze input data characteristics for prompt optimization."""
        return {
            "length": len(input_data),
            "complexity": self._estimate_complexity(input_data),
            "format": self._detect_format(input_data)
        }
        
    def _customize_template(self, template: str, features: Dict, 
                          additional_context: Dict) -> str:
        """Customize template based on input features and context."""
        # Implement template customization logic
        customized = template
        
        if features["length"] > 1000:
            customized = "Please provide a concise response. " + customized
            
        if additional_context:
            customized = self._incorporate_context(customized, additional_context)
            
        return customized
