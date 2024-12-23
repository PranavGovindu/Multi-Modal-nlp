import optuna
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader
import torch
import faiss
from sentence_transformers import SentenceTransformer
import spacy
from neo4j import GraphDatabase
import nltk
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
        
logger = logging.getLogger(__name__)

def process_and_store_documents(documents):
    if not documents:
        raise ValueError("Documents list cannot be empty")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
        def preprocess_text(text):
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [token for token in tokens 
                     if token not in stop_words
                     and token not in string.punctuation
                     and token.isalpha()]
            return ' '.join(tokens)
        
        processed_docs = [preprocess_text(doc) for doc in documents]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(processed_docs)
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        logger.info(f"Successfully processed and indexed {len(documents)} documents")
        return index, processed_docs, embeddings
    except Exception as e:
        logger.error(f"Failed to process and store documents: {str(e)}")
        raise RuntimeError(f"Document processing failed: {str(e)}")


class ModelOptimizer:
    TASK_CONFIGS = {
        'classification': {
            'model_types': ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'],
            'metrics': ['accuracy', 'f1'],
            'model_class': AutoModelForSequenceClassification,
        },
        'summarization': {
            'model_types': ['t5-small', 'facebook/bart-base', 'google/pegasus-small'],
            'metrics': ['rouge'],
            'model_class': AutoModelForSeq2SeqLM,
        },
        'question_answering': {
            'model_types': ['distilbert-base-uncased', 'bert-base-uncased'],
            'metrics': ['exact_match', 'f1'],
            'model_class': AutoModelForQuestionAnswering,
        }
    }

    def __init__(self, task: str, dataset: Union[Dataset, Dict], num_trials: int = 20):
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Task must be one of {list(self.TASK_CONFIGS.keys())}")
            
        self.task = task
        self.dataset = dataset
        self.num_trials = num_trials
        self.config = self.TASK_CONFIGS[task]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics = {
            metric: load_metric(metric) for metric in self.config['metrics']
        }

    def preprocess_dataset(self, tokenizer: AutoTokenizer) -> Dataset:
        try:
            if self.task == 'classification':
                def tokenize_function(examples):
                    return tokenizer(
                        examples['text'],
                        truncation=True,
                        padding='max_length',
                        max_length=512
                    )
                
            elif self.task == 'summarization':
                def tokenize_function(examples):
                    model_inputs = tokenizer(
                        examples['text'],
                        truncation=True,
                        padding='max_length',
                        max_length=512
                    )
                    labels = tokenizer(
                        examples['summary'],
                        truncation=True,
                        padding='max_length',
                        max_length=128
                    )
                    model_inputs['labels'] = labels['input_ids']
                    return model_inputs
            
            elif self.task == 'question_answering':
                def tokenize_function(examples):
                    return tokenizer(
                        examples['question'],
                        examples['context'],
                        truncation=True,
                        padding='max_length',
                        max_length=512,
                        return_offsets_mapping=True
                    )

            tokenized_dataset = self.dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.dataset.column_names
            )

            return tokenized_dataset

        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        if self.task == 'classification':
            predictions = np.argmax(predictions, axis=1)
            return {
                name: metric.compute(predictions=predictions, references=labels)
                for name, metric in self.metrics.items()
            }
        
        elif self.task == 'summarization':
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            return self.metrics['rouge'].compute(predictions=decoded_preds, references=decoded_labels)
        
        elif self.task == 'question_answering':
            return {
                name: metric.compute(predictions=predictions, references=labels)
                for name, metric in self.metrics.items()
            }

    def objective(self, trial: optuna.Trial) -> float:
        try:
            model_name = trial.suggest_categorical('model_name', self.config['model_types'])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            num_epochs = trial.suggest_int('num_epochs', 2, 5)
            weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = self.config['model_class'].from_pretrained(
                model_name,
                num_labels=2 if self.task == 'classification' else None
            ).to(self.device)
            
            processed_dataset = self.preprocess_dataset(self.tokenizer)
            train_dataset, eval_dataset = train_test_split(processed_dataset, test_size=0.2)
            
            training_args = TrainingArguments(
                output_dir=f'./results/{trial.number}',
                evaluation_strategy='epoch',
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=weight_decay,
                logging_dir=f'./logs/{trial.number}',
                logging_steps=100,
                save_strategy='epoch',
                load_best_model_at_end=True,
            )
            
            data_collator = (
                DataCollatorForSeq2Seq(self.tokenizer)
                if self.task == 'summarization'
                else DataCollatorWithPadding(self.tokenizer)
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics
            )
            
            trainer.train()
            eval_results = trainer.evaluate()
            
            main_metric = list(self.metrics.keys())[0]
            metric_value = eval_results[f'eval_{main_metric}']
            
            logger.info(f"Trial {trial.number}: {main_metric} = {metric_value}")
            return metric_value

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            raise optuna.TrialPruned()

    def optimize(self) -> Dict[str, Any]:
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.num_trials)
        
        best_trial = study.best_trial
        logger.info(f"Best trial:")
        logger.info(f"  Value: {best_trial.value}")
        logger.info(f"  Params: {best_trial.params}")
        
        return {
            'best_params': best_trial.params,
            'best_value': best_trial.value,
            'best_model_name': best_trial.params['model_name'],
            'study': study
        }

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")

    def close(self):
        self.driver.close()

    def build_knowledge_graph(self, text):
        doc = self.nlp(text)
        with self.driver.session() as session:
            try:
                for ent in doc.ents:
                    with session.begin_transaction() as tx:
                        self._create_and_link_entity(tx, ent.text, ent.label_)
            except Exception as e:
                logger.error(f"Failed to build knowledge graph: {str(e)}")
                raise
    @staticmethod
    def _create_and_link_entity(tx, entity, label):
        query = """
        MERGE (e:Entity {name: $entity}) 
        ON CREATE SET e.label = $label
        """
        tx.run(query, entity=entity, label=label)
        
    def knowledge_graph_search(self, query):
        with self.driver.session() as session:
            result = session.read_transaction(self._search_entity, query)
            return [record["e.name"] for record in result]

    @staticmethod
    def _search_entity(tx, query):
        return tx.run("MATCH (e:Entity) "
                      "WHERE e.name CONTAINS $query "
                      "RETURN e.name", query=query)
