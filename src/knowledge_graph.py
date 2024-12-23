import spacy
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import RDFS, Namespace
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.graph = Graph()
        self.ns = Namespace("http://example.org/")
        
    def build_knowledge_graph(self, text):
        """Convert raw text into a knowledge graph with semantic relationships."""
        try:
            doc = self.nlp(text)
            
            # Process entities and their relationships
            for sent in doc.sents:
                processed_sent = self.nlp(sent.text)
                self._extract_entities_and_relations(processed_sent)
                
            logger.info(f"Successfully built knowledge graph with {len(self.graph)} triples")
            return self.graph
            
        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {str(e)}")
            raise
            
    def _extract_entities_and_relations(self, doc):
        """Extract entities and their relationships from processed text."""
        for ent in doc.ents:
            subject = URIRef(self.ns[ent.text.replace(" ", "_")])
            self.graph.add((subject, RDF.type, self.ns[ent.label_]))
            
        # Extract relationships from dependency parse
        for token in doc:
            if token.dep_ in ("nsubj", "dobj"):
                self._add_relationship(token)
                
    def _add_relationship(self, token):
        """Add relationship triple to the graph."""
        if token.head.pos_ == "VERB":
            subject = URIRef(self.ns[token.text.replace(" ", "_")])
            predicate = URIRef(self.ns[token.head.lemma_])
            obj = URIRef(self.ns[token.head.text.replace(" ", "_")])
            self.graph.add((subject, predicate, obj))
