import pandas as pd
import os


class DataIngestion:
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'artificats', 'raw', 'sentimentdataset.csv')
        self.data_path = data_path
        self.df = None
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
    
    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        self.df.columns = self.df.columns.str.strip()
        
        text_col = self._detect_column('Text', ['text', 'Text', 'tweet', 'content'])
        sentiment_col = self._detect_column('Sentiment', ['sentiment', 'Sentiment', 'label', 'class'])
        platform_col = self._detect_column('Platform', ['platform', 'Platform', 'source'])
        
        rename_map = {}
        if text_col != 'Text':
            rename_map[text_col] = 'Text'
        if sentiment_col != 'Sentiment':
            rename_map[sentiment_col] = 'Sentiment'
        if platform_col and platform_col != 'Platform':
            rename_map[platform_col] = 'Platform'
        
        if rename_map:
            self.df = self.df.rename(columns=rename_map)
        
        self.df['Sentiment'] = self.df['Sentiment'].str.strip()
        self.df['Sentiment'] = self.df['Sentiment'].apply(self._map_sentiment)
        
        print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Sentiment distribution:\n{self.df['Sentiment'].value_counts()}")
        
        return self.df

    def _map_sentiment(self, sentiment: str) -> str:
        sentiment_lower = str(sentiment).lower().strip()
        if sentiment_lower in self.sentiment_mapping:
            return self.sentiment_mapping[sentiment_lower]
        for key, value in self.sentiment_mapping.items():
            if key in sentiment_lower:
                return value
        return 'Positive' if sentiment_lower not in ['negative', 'neutral'] else sentiment_lower.capitalize()
    
    def _detect_column(self, preferred: str, alternatives: list) -> str:
        if preferred in self.df.columns:
            return preferred
        for col in alternatives:
            if col in self.df.columns:
                return col
        return self.df.columns[0]

    def get_data(self) -> pd.DataFrame:
        if self.df is None:
            return self.load_data()
        return self.df


if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
    print(df.head())