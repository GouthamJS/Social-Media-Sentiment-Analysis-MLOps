import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import DataIngestion
from eda import EDA
from preprocessing import TextPreprocessing
from feature_engineering import FeatureEngineering
from model_trainer import ModelTrainer
from model_evaluation import ModelEvaluation


class TrainingPipeline:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.model = None
        self.evaluator = None

    def run(self):
        print("=" * 60)
        print("Starting Training Pipeline")
        print("=" * 60)

        print("\n[Step 1/6] Data Ingestion")
        print("-" * 40)
        ingestion = DataIngestion()
        self.df = ingestion.load_data()

        print("\n[Step 2/6] Exploratory Data Analysis")
        print("-" * 40)
        eda = EDA(self.df)
        eda.generate_all()

        print("\n[Step 3/6] Text Preprocessing")
        print("-" * 40)
        preprocessor = TextPreprocessing()
        self.df = preprocessor.process(self.df)

        print("\n[Step 4/6] Feature Engineering (TF-IDF)")
        print("-" * 40)
        feature_eng = FeatureEngineering()
        X = feature_eng.fit_transform(self.df, text_column='clean_text')
        feature_eng.save_vectorizer()

        print("\n[Step 5/6] Model Training")
        print("-" * 40)
        y = self.df['Sentiment']
        trainer = ModelTrainer(test_size=0.2, random_state=42)
        trainer.prepare_data(X, y)
        trainer.train()
        trainer.save_model()

        print("\n[Step 6/6] Model Evaluation")
        print("-" * 40)
        y_pred = trainer.predict(trainer.X_test)
        y_pred_numeric = np.array([trainer.label_encoder[p] for p in y_pred])
        self.evaluator = ModelEvaluation()
        metrics = self.evaluator.evaluate(
            trainer.y_test, 
            y_pred_numeric, 
            labels=list(trainer.label_encoder.keys())
        )
        self.evaluator.save_metrics(metrics)

        print("\n" + "=" * 60)
        print("Training Pipeline Completed Successfully!")
        print("=" * 60)
        
        return {
            'model': trainer.model,
            'vectorizer': feature_eng.vectorizer,
            'metrics': metrics,
            'label_encoder': trainer.label_encoder
        }


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    result = pipeline.run()
    print("\nFinal Results:")
    print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {result['metrics']['f1_score']:.4f}")