import pandas as pd
import re
import os


class TextPreprocessing:
    def __init__(self):
        self.text_column = 'Text'
        self.output_column = 'clean_text'
        self.sentiment_mapping = {
            'positive': 'Positive', 'joy': 'Positive', 'excitement': 'Positive',
            'contentment': 'Positive', 'love': 'Positive', 'admiration': 'Positive',
            'optimism': 'Positive', 'serenity': 'Positive', 'joyfulness': 'Positive',
            'enthusiasm': 'Positive', 'delight': 'Positive', 'pride': 'Positive',
            'amusement': 'Positive', 'relief': 'Positive', 'satisfaction': 'Positive',
            'grateful': 'Positive', 'hope': 'Positive', 'inspiration': 'Positive',
            'amazement': 'Positive', 'delightedness': 'Positive', 'ecstasy': 'Positive',
            'elation': 'Positive', 'euphoria': 'Positive', 'glee': 'Positive',
            'happiness': 'Positive', 'pleasure': 'Positive', 'bliss': 'Positive',
            'cheerfulness': 'Positive', 'contentedness': 'Positive', 'exhilaration': 'Positive',
            'positive vibes': 'Positive', 'celebration': 'Positive', 'love it': 'Positive',
            'happy': 'Positive', 'great': 'Positive', 'awesome': 'Positive', 'amazing': 'Positive',
            'wonder': 'Positive', 'awe': 'Positive', 'celestial wonder': 'Positive',
            "nature's beauty": 'Positive', 'thrilling journey': 'Positive',
            'negative': 'Negative', 'sadness': 'Negative', 'anger': 'Negative',
            'fear': 'Negative', 'disgust': 'Negative', 'frustration': 'Negative',
            'disappointment': 'Negative', 'loneliness': 'Negative', 'anxiety': 'Negative',
            'jealousy': 'Negative', 'envy': 'Negative', 'regret': 'Negative',
            'guilt': 'Negative', 'shame': 'Negative', 'embarrassment': 'Negative',
            'sorrow': 'Negative', 'grief': 'Negative', 'depression': 'Negative',
            'melancholy': 'Negative', 'pessimism': 'Negative', 'hopelessness': 'Negative',
            'worried': 'Negative', 'upset': 'Negative', 'terrible': 'Negative',
            'horrible': 'Negative', 'hate': 'Negative', 'bad': 'Negative', 'awful': 'Negative',
            'neutral': 'Neutral', 'indifference': 'Neutral', 'boredom': 'Neutral',
            'curiosity': 'Neutral', 'confusion': 'Neutral', 'surprise': 'Neutral',
            'contemplation': 'Neutral', 'reflection': 'Neutral', 'thoughtful': 'Neutral',
            'ok': 'Neutral', 'okay': 'Neutral', 'fine': 'Neutral', 'meh': 'Neutral'
        }
    
    def map_sentiment(self, sentiment: str) -> str:
        sentiment_lower = str(sentiment).lower().strip()
        if sentiment_lower in self.sentiment_mapping:
            return self.sentiment_mapping[sentiment_lower]
        for key, value in self.sentiment_mapping.items():
            if key in sentiment_lower or sentiment_lower in key:
                return value
        return 'Positive' if sentiment_lower not in ['negative', 'neutral'] else sentiment_lower.capitalize()

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'@\w+', '', text)
        
        text = re.sub(r'#(\w+)', r'\1', text)
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in DataFrame")
        
        df[self.output_column] = df[self.text_column].apply(self.clean_text)
        
        df['text_length'] = df[self.output_column].str.len()
        
        print(f"Text preprocessing completed")
        print(f"  - Sample cleaned texts (first 3):")
        for i in range(min(3, len(df))):
            orig = str(df[self.text_column].iloc[i])[:50].encode('ascii', 'replace').decode('ascii')
            clean = str(df[self.output_column].iloc[i])[:50].encode('ascii', 'replace').decode('ascii')
            print(f"    Original: {orig}...")
            print(f"    Cleaned:  {clean}...")
        print()
        
        return df

    def get_cleaned_column(self) -> str:
        return self.output_column


if __name__ == "__main__":
    from data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    preprocessor = TextPreprocessing()
    df = preprocessor.process(df)
    
    print(df[['Text', 'clean_text']].head())