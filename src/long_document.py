from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
import logging

logger = logging.getLogger(__name__)

class LongDocumentProcessor:
    def __init__(self):
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.model = LongformerForSequenceClassification.from_pretrained(
            'allenai/longformer-base-4096'
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def summarize_long_document_with_longformer(self, doc):
        """Summarize long documents using Longformer."""
        try:
            # Check document length
            if len(doc.split()) < 100:
                return doc
                
            # Process document in chunks
            chunks = self._split_into_chunks(doc)
            summaries = []
            
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors='pt',
                    max_length=4096,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=150,
                    min_length=40,
                    num_beams=4
                )
                
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                summaries.append(summary)
                
            final_summary = ' '.join(summaries)
            logger.info("Successfully summarized long document")
            return final_summary
            
        except Exception as e:
            logger.error(f"Failed to summarize document: {str(e)}")
            raise
            
    def _split_into_chunks(self, doc, max_chunk_size=4000):
        """Split document into manageable chunks."""
        words = doc.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
