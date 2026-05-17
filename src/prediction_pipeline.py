import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
from preprocessing import TextPreprocessing
from feature_engineering import FeatureEngineering


class PredictionPipeline:
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'artifacts', 'model.pkl')
        
        self.model_path = model_path
        self.model = None
        self.bert_engineer = None
        self.label_encoder = None
        self.reverse_encoder = None
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.reverse_encoder = model_data['reverse_encoder']
        
        self.bert_engineer = FeatureEngineering()
        
        print(f"Model loaded from: {self.model_path}")
        print(f"Using BERT embeddings (768 features)")
        print(f"Labels: {self.label_encoder}")
        
        return self
        
    def predict(self, text: str):
        if self.model is None:
            self.load_model()
        
        preprocessor = TextPreprocessing()
        cleaned_text = preprocessor.clean_text(text)
        
        import pandas as pd
        df = pd.DataFrame({'clean_text': [cleaned_text]})
        X = self.bert_engineer.transform(df, 'clean_text')
        
        prediction = self.model.predict(X)[0]
        sentiment = self.reverse_encoder[prediction]
        
        probabilities = self.model.predict_proba(X)[0]
        prob_dict = {self.reverse_encoder[i]: round(float(prob), 4) for i, prob in enumerate(probabilities)}
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'probabilities': prob_dict
        }
        
    def predict_batch(self, texts: list):
        if self.model is None:
            self.load_model()
        
        preprocessor = TextPreprocessing()
        cleaned_texts = [preprocessor.clean_text(text) for text in texts]
        
        import pandas as pd
        df = pd.DataFrame({'clean_text': cleaned_texts})
        X = self.bert_engineer.transform(df, 'clean_text')
        
        predictions = self.model.predict(X)
        sentiments = [self.reverse_encoder[p] for p in predictions]
        
        return texts, sentiments


if __name__ == "__main__":
    pipeline = PredictionPipeline()
    
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "The weather is okay today.",
        "So excited for the weekend!",
        "Very disappointed with the service."
    ]
    
    print("Testing Prediction Pipeline:")
    print("=" * 50)
    
    for text in test_texts:
        result = pipeline.predict(text)
        print(f"\nText: {result['text']}")
        print(f"Cleaned: {result['cleaned_text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Probabilities: {result['probabilities']}")