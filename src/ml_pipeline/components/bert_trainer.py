import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import re


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTTrainer:
    def __init__(self, model_name='bert-base-uncased', max_length=128, batch_size=16, epochs=3, learning_rate=2e-5):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3
        )
        self.model.to(self.device)
        
        self.label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        self.sentiment_mapping = {
            'positive': 'Positive', 'joy': 'Positive', 'excitement': 'Positive',
            'contentment': 'Positive', 'love': 'Positive', 'admiration': 'Positive',
            'optimism': 'Positive', 'serenity': 'Positive', 'joyfulness': 'Positive',
            'enthusiasm': 'Positive', 'delight': 'Positive', 'pride': 'Positive',
            'amusement': 'Positive', 'relief': 'Positive', 'satisfaction': 'Positive',
            'grateful': 'Positive', 'hope': 'Positive', 'inspiration': 'Positive',
            'amazement': 'Positive', 'happiness': 'Positive', 'happy': 'Positive',
            'great': 'Positive', 'awesome': 'Positive', 'amazing': 'Positive',
            'wonder': 'Positive', 'awe': 'Positive', 'celestial wonder': 'Positive',
            "nature's beauty": 'Positive', 'thrilling journey': 'Positive',
            'negative': 'Negative', 'sadness': 'Negative', 'anger': 'Negative',
            'fear': 'Negative', 'disgust': 'Negative', 'frustration': 'Negative',
            'disappointment': 'Negative', 'loneliness': 'Negative', 'anxiety': 'Negative',
            'jealousy': 'Negative', 'envy': 'Negative', 'terrible': 'Negative',
            'hate': 'Negative', 'bad': 'Negative', 'awful': 'Negative',
            'neutral': 'Neutral', 'boredom': 'Neutral', 'ok': 'Neutral', 'okay': 'Neutral',
            'fine': 'Neutral', 'curiosity': 'Neutral', 'confusion': 'Neutral'
        }
    
    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        if 'Text' not in df.columns:
            raise ValueError("Text column not found in dataset")
        if 'Sentiment' not in df.columns:
            raise ValueError("Sentiment column not found in dataset")
        
        df['Sentiment'] = df['Sentiment'].str.strip().apply(self._map_sentiment)
        
        texts = df['Text'].apply(self._clean_text).values
        labels = df['Sentiment'].map(self.label_mapping).values
        
        print(f"Loaded {len(texts)} samples")
        print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels
    
    def _map_sentiment(self, sentiment):
        sentiment_lower = str(sentiment).lower().strip()
        if sentiment_lower in self.sentiment_mapping:
            return self.sentiment_mapping[sentiment_lower]
        for key, value in self.sentiment_mapping.items():
            if key in sentiment_lower:
                return value
        return 'Positive' if sentiment_lower not in ['negative', 'neutral'] else sentiment_lower.capitalize()
    
    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_data(self, texts, labels, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = SentimentDataset(X_train, y_train, self.tokenizer, self.max_length)
        test_dataset = SentimentDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        total_steps = len(self.train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        self.model.train()
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            total_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.model.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Average training loss: {avg_loss:.4f}")
    
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        print("\nEvaluating on test set...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"\n{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print(f"{'='*50}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive']))
        
        return {'accuracy': accuracy, 'f1_score': f1}
    
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        mapping_path = os.path.join(save_path, 'label_mapping.pt')
        torch.save({'label_mapping': self.label_mapping, 'reverse_mapping': self.reverse_mapping}, mapping_path)
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, save_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(save_path)
        self.tokenizer = AutoTokenizer.from_pretrained(save_path)
        self.model.to(self.device)
        
        mapping_path = os.path.join(save_path, 'label_mapping.pt')
        if os.path.exists(mapping_path):
            mapping = torch.load(mapping_path)
            self.label_mapping = mapping['label_mapping']
            self.reverse_mapping = mapping['reverse_mapping']
        
        print(f"Model loaded from: {save_path}")
    
    def predict(self, text):
        self.model.eval()
        
        cleaned_text = self._clean_text(text)
        
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        
        sentiment = self.reverse_mapping[pred]
        prob_dict = {self.reverse_mapping[i]: round(probs[0][i].item(), 4) for i in range(3)}
        
        return {'text': text, 'sentiment': sentiment, 'probabilities': prob_dict}


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_path = os.path.join(base_path, 'artificats', 'raw', 'sentimentdataset.csv')
    save_path = os.path.join(base_path, 'artifacts', 'bert_model')
    
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")
    
    trainer = BERTTrainer(
        model_name='bert-base-uncased',
        max_length=64,
        batch_size=8,
        epochs=1,
        learning_rate=3e-5
    )
    
    texts, labels = trainer.load_data(data_path)
    trainer.prepare_data(texts, labels, test_size=0.2)
    trainer.train()
    
    metrics = trainer.evaluate()
    trainer.save_model(save_path)
    
    print("\nPrediction test:")
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "The weather is okay today."
    ]
    for text in test_texts:
        result = trainer.predict(text)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Probabilities: {result['probabilities']}\n")