from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = None
        
    def semantic_search_with_sentence_transformers(self, query, top_k=5):
        """Perform semantic search using sentence transformers and FAISS."""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            query_embedding = np.float32(query_embedding)
            
            if self.index is None:
                raise ValueError("Vector database not initialized")
                
            # Perform similarity search
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Prepare results
            results = [
                {
                    'document': self.documents[idx],
                    'score': float(score)
                }
                for score, idx in zip(distances[0], indices[0])
            ]
            
            logger.info(f"Successfully performed semantic search for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {str(e)}")
            raise
