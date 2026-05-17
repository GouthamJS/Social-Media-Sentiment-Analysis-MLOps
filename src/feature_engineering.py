import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import joblib
import os


class FeatureEngineering:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 768
        self.X = None

    def fit_transform(self, df: pd.DataFrame, text_column: str = 'clean_text') -> np.ndarray:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].fillna('').tolist()
        
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            self.X = outputs.last_hidden_state[:, 0, :].numpy()
        
        print(f"BERT Embeddings completed")
        print(f"  - Embedding dimensions: {self.embedding_dim}")
        print(f"  - Feature matrix shape: {self.X.shape}")
        
        return self.X

    def transform(self, df: pd.DataFrame, text_column: str = 'clean_text') -> np.ndarray:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].fillna('').tolist()
        
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            X = outputs.last_hidden_state[:, 0, :].numpy()
        
        return X

    def save_vectorizer(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'artifacts', 'bert_tokenizer.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model_name': self.model_name, 'embedding_dim': self.embedding_dim}, path)
        print(f"BERT Tokenizer config saved to: {path}")

    def load_vectorizer(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'artifacts', 'bert_tokenizer.pkl')
        config = joblib.load(path)
        self.model_name = config['model_name']
        self.embedding_dim = config['embedding_dim']
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.eval()
        print(f"BERT Tokenizer loaded from: {path}")
        return self.tokenizer


if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from preprocessing import TextPreprocessing
    
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    preprocessor = TextPreprocessing()
    df = preprocessor.process(df)
    
    feature_eng = FeatureEngineering('bert-base-uncased')
    X = feature_eng.fit_transform(df, 'clean_text')
    feature_eng.save_vectorizer()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Embedding dimensions: {feature_eng.embedding_dim}")