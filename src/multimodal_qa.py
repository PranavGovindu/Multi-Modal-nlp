import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class MultiModalQA:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        
    def multi_modal_qa(self, image, text, question):
        """Process both image and text inputs to answer questions."""
        try:
            # Process image
            if isinstance(image, str):
                image = Image.open(image)
            
            # Prepare inputs
            inputs = self.processor(
                text=[question],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get model outputs
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Compute similarity and generate answer
            similarity = torch.nn.functional.cosine_similarity(
                image_features, text_features
            )
            
            # Generate answer based on similarity and context
            answer = self._generate_answer(similarity, question, text)
            
            logger.info("Successfully processed multi-modal query")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to process multi-modal query: {str(e)}")
            raise
            
    def _generate_answer(self, similarity, question, context):
        """Generate answer based on similarity scores and context."""
        # Implement answer generation logic based on similarity scores
        threshold = 0.5
        if similarity.mean().item() > threshold:
            return self._extract_answer_from_context(question, context)
        return "Unable to find a relevant answer."
