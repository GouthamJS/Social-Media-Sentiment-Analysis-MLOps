import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os


class ModelEvaluation:
    def __init__(self):
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.conf_matrix = None
        self.class_report = None

    def evaluate(self, y_true, y_pred, labels=None):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        if y_true_arr.dtype in ['int64', 'int32'] and y_pred_arr.dtype == 'object':
            y_pred_arr = np.array([self._get_label_encoding(pred, labels) for pred in y_pred_arr])
        
        self.accuracy = accuracy_score(y_true_arr, y_pred_arr)
        self.precision = precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        self.recall = recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        self.f1 = f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        
        self.conf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
        self.class_report = classification_report(y_true_arr, y_pred_arr, target_names=labels, zero_division=0)
        
        print("Model Evaluation Results:")
        print("=" * 50)
        print(f"Accuracy:  {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"F1 Score:  {self.f1:.4f}")
        print("=" * 50)
        print("\nClassification Report:")
        print(self.class_report)
        
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1,
            'confusion_matrix': self.conf_matrix.tolist(),
            'classification_report': self.class_report
        }

    def _get_label_encoding(self, pred_label, labels):
        if labels and pred_label in labels:
            return list(labels).index(pred_label)
        return pred_label

    def get_metrics(self) -> dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1
        }

    def save_metrics(self, metrics: dict, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'artifacts', 'metrics.txt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
            f.write("=" * 50 + "\n")
            f.write("\nClassification Report:\n")
            f.write(metrics['classification_report'])
        
        print(f"Metrics saved to: {path}")


if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from preprocessing import TextPreprocessing
    from feature_engineering import FeatureEngineering
    from model_trainer import ModelTrainer
    
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
    
    y_pred = trainer.predict(trainer.X_test)
    
    evaluator = ModelEvaluation()
    metrics = evaluator.evaluate(trainer.y_test, y_pred, labels=list(trainer.label_encoder.keys()))
    evaluator.save_metrics(metrics)