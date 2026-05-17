import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os


class ModelTrainer:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            C=1.0,
            multi_class='multinomial',
            solver='lbfgs'
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None

    def prepare_data(self, X, y):
        self.label_encoder = {label: idx for idx, label in enumerate(sorted(y.unique()))}
        self.reverse_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        y_encoded = y.map(self.label_encoder)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        print(f"Data split completed")
        print(f"  - Training samples: {self.X_train.shape[0]}")
        print(f"  - Test samples: {self.X_test.shape[0]}")
        print(f"  - Labels: {self.label_encoder}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print("Training Logistic Regression model...")
        self.model.fit(self.X_train, self.y_train)
        
        train_score = self.model.score(self.X_train, self.y_train)
        print(f"Training accuracy: {train_score:.4f}")
        
        return self.model

    def save_model(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'artifacts', 'model.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'reverse_encoder': self.reverse_encoder
        }
        joblib.dump(model_data, path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'artifacts', 'model.pkl')
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.reverse_encoder = model_data['reverse_encoder']
        print(f"Model loaded from: {path}")
        return self.model

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.array([self.reverse_encoder[p] for p in predictions])

    def predict_proba(self, X):
        probas = self.model.predict_proba(X)
        return probas


if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from preprocessing import TextPreprocessing
    from feature_engineering import FeatureEngineering
    
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    preprocessor = TextPreprocessing()
    df = preprocessor.process(df)
    
    feature_eng = FeatureEngineering()
    X = feature_eng.fit_transform(df)
    y = df['Sentiment']
    
    trainer = ModelTrainer()
    trainer.prepare_data(X, y)
    trainer.train()
    trainer.save_model()