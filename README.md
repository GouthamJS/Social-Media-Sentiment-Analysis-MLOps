# Social Media Sentiment Analysis MLOps Project

A production-level ML pipeline for analyzing sentiment in social media posts.

## Project Structure
```bash
project/
├── artifacts/
│   ├── raw/
│   │   └── sentimentdataset.csv
│   ├── eda/
│   │   ├── sentiment_distribution.png
│   │   ├── text_length_analysis.png
│   │   └── platform_sentiment.png
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── metrics.txt
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   ├── training_pipeline.py
│   └── prediction_pipeline.py
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Features

- **Data Ingestion**: Load and preprocess CSV data
- **EDA**: Sentiment distribution, text length analysis, platform insights
- **Text Preprocessing**: Clean text, remove URLs, mentions, and special characters
- **Feature Engineering**: TF-IDF vectorization with n-grams
- **Model Training**: Logistic Regression with balanced class weights
- **Model Evaluation**: Accuracy, Precision, Recall, and F1 Score
- **Experiment Tracking** using MLflow
- **Model Versioning** using DagsHub
- **Docker Containerization**
- **AWS EKS Deployment**
- **Monitoring** using Prometheus & Grafana

## Dataset

- **Input Feature**: `Text`
- **Target Variable**: `Sentiment`
- **Classes**:
  - Positive
  - Negative
  - Neutral
- **Platforms**:
  - Twitter
  - Instagram
  - Facebook

## Workflow

1. Data ingestion and preprocessing
2. Feature extraction using TF-IDF
3. Model training and evaluation
4. Experiment tracking with MLflow
5. Model versioning using DagsHub
6. Docker containerization
7. Deployment on AWS EKS
8. Monitoring using Prometheus & Grafana

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Create Conda Environment
```bash
conda create -n mlops_env python=3.9
conda activate mlops_env
```

### Run Training Pipeline
```bash
python -m src.training_pipeline
```

### Test Predictions
```bash
python -m src.prediction_pipeline
```

### Start Flask API
```bash
python app.py
```

## API Endpoints

### POST /predict
Predict sentiment from input text.

```json
{
  "text": "I love this product!"
}
```

### GET /dashboard
Returns dataset statistics and model insights.

## Deployment

### Docker
Containerized the application for portability and environment consistency.

### AWS EKS
Deployed the containerized application on Kubernetes using AWS EKS for:
- Scalability
- High availability
- Load balancing

## Monitoring

Implemented monitoring using:
- Prometheus
- Grafana

Tracked metrics:
- CPU usage
- Memory usage
- API response time
- Request rate

## Results

- Achieved accurate sentiment classification
- Automated complete ML workflow
- Enabled scalable deployment
- Implemented real-time monitoring

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Flask
- MLflow
- DagsHub
- Docker
- AWS EKS
- Prometheus
- Grafana

## Results

- Successfully built an end-to-end MLOps pipeline for sentiment analysis
- Performed sentiment classification on social media text with reliable accuracy
- Automated data preprocessing, model training, evaluation, and deployment workflows
- Tracked experiments and model versions using MLflow and DagsHub
- Deployed the application on AWS EKS using Docker and Kubernetes
- Implemented real-time monitoring using Prometheus and Grafana
- Enabled scalable and production-ready sentiment prediction through a web application

## Future Improvements

- Add BERT-based sentiment analysis
- Implement drift detection
- Support real-time streaming data
- Improve dashboard UI/UX
- Add multilingual support