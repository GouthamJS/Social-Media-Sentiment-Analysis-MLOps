import pandas as pd
import numpy as np
import time
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


class DashboardService:
    def __init__(self, data_path: str = None):
        if data_path is None:
            import os
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'artificats', 'raw', 'sentimentdataset.csv')
        self.data_path = data_path
        self._cache = None
        self._cache_time = None
        self.CACHE_TTL = 300
        self._df = None

    def _load_data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.data_path)
            self._df.columns = self._df.columns.str.strip()
            
            sentiment_mapping = {
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
            
            if 'Sentiment' in self._df.columns:
                self._df['Sentiment'] = self._df['Sentiment'].str.strip()
                self._df['Sentiment'] = self._df['Sentiment'].apply(lambda x: self._map_sentiment(x, sentiment_mapping))
            
            if 'Text' in self._df.columns:
                self._df['clean_text'] = self._df['Text'].apply(self._clean_text)
                self._df['text_length'] = self._df['clean_text'].str.len()
        
        return self._df

    def _clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _map_sentiment(self, sentiment: str, mapping: dict) -> str:
        sentiment_lower = str(sentiment).lower().strip()
        if sentiment_lower in mapping:
            return mapping[sentiment_lower]
        for key, value in mapping.items():
            if key in sentiment_lower:
                return value
        return 'Positive' if sentiment_lower not in ['negative', 'neutral'] else sentiment_lower.capitalize()

    def get_dashboard_data(self, force_refresh: bool = False) -> dict:
        if self._cache is not None and not force_refresh:
            if time.time() - self._cache_time < self.CACHE_TTL:
                return self._cache
        
        self._cache = self._compute_dashboard()
        self._cache_time = time.time()
        return self._cache

    def _compute_dashboard(self) -> dict:
        df = self._load_data()
        
        return {
            'kpis': self._get_kpis(df),
            'text_analysis': self._get_text_analysis(df),
            'platform_analysis': self._get_platform_insights(df),
            'time_analysis': self._get_time_insights(df),
            'top_keywords': self._get_top_keywords(df),
            'top_hashtags': self._get_hashtag_insights(df),
            'country_analysis': self._get_country_analysis(df)
        }

    def _get_kpis(self, df: pd.DataFrame) -> dict:
        total = len(df)
        sentiment_dist = df['Sentiment'].value_counts()
        
        positive = int(sentiment_dist.get('Positive', 0))
        negative = int(sentiment_dist.get('Negative', 0))
        neutral = int(sentiment_dist.get('Neutral', 0))
        
        return {
            'total_records': total,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'positive_percentage': round(positive / total * 100, 2) if total > 0 else 0,
            'negative_percentage': round(negative / total * 100, 2) if total > 0 else 0,
            'neutral_percentage': round(neutral / total * 100, 2) if total > 0 else 0
        }

    def _get_text_analysis(self, df: pd.DataFrame) -> dict:
        if 'text_length' not in df.columns:
            return {'average_text_length': 0, 'max_text_length': 0, 'min_text_length': 0}
        
        return {
            'average_text_length': round(df['text_length'].mean(), 2),
            'max_text_length': int(df['text_length'].max()),
            'min_text_length': int(df['text_length'].min())
        }

    def _get_platform_insights(self, df: pd.DataFrame) -> dict:
        if 'Platform' not in df.columns:
            return {'available': False}
        
        distributions = {}
        platform_totals = df['Platform'].value_counts()
        
        for platform in platform_totals.index:
            platform_df = df[df['Platform'] == platform]
            sentiment_counts = platform_df['Sentiment'].value_counts()
            distributions[platform.strip()] = {
                'total': int(platform_totals[platform]),
                'Positive': int(sentiment_counts.get('Positive', 0)),
                'Negative': int(sentiment_counts.get('Negative', 0)),
                'Neutral': int(sentiment_counts.get('Neutral', 0))
            }
        
        top_platform = platform_totals.index[0] if len(platform_totals) > 0 else None
        top_count = int(platform_totals.iloc[0]) if len(platform_totals) > 0 else 0
        
        return {
            'available': True,
            'distributions': distributions,
            'top_platform': top_platform.strip() if top_platform else None,
            'top_platform_count': top_count
        }

    def _get_time_insights(self, df: pd.DataFrame) -> dict:
        result = {'available': False, 'yearly_trend': None}
        
        if 'Year' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['Year'] = df_copy['Year'].astype(int)
                
                yearly_trend_df = df_copy.groupby(['Year', 'Sentiment']).size().unstack(fill_value=0)
                
                yearly_sentiment_trend = []
                for year in sorted(yearly_trend_df.index):
                    row = {'year': int(year)}
                    for sent in ['Positive', 'Negative', 'Neutral']:
                        row[sent] = int(yearly_trend_df.loc[year].get(sent, 0))
                    yearly_sentiment_trend.append(row)
                
                result['available'] = True
                result['yearly_trend'] = yearly_sentiment_trend
            except:
                pass
        
        if 'Year' in df.columns and 'Month' in df.columns and 'Day' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['date_str'] = df_copy['Year'].astype(str) + '-' + \
                    df_copy['Month'].astype(str).str.zfill(2) + '-' + \
                    df_copy['Day'].astype(str).str.zfill(2)
                
                trend = df_copy.groupby(['date_str', 'Sentiment']).size().unstack(fill_value=0)
                
                sentiment_trend = []
                for date in trend.index:
                    row = {'date': date}
                    for sent in ['Positive', 'Negative', 'Neutral']:
                        row[sent] = int(trend.loc[date].get(sent, 0))
                    sentiment_trend.append(row)
                
                result['sentiment_trend'] = sentiment_trend[:30]
            except:
                pass
        
        return result

    def _get_top_keywords(self, df: pd.DataFrame, top_n: int = 10) -> list:
        if 'clean_text' not in df.columns:
            return []
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 1)
            )
            tfidf_matrix = vectorizer.fit_transform(df['clean_text'].fillna(''))
            
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores[:top_n]]
        except:
            return []

    def _get_hashtag_insights(self, df: pd.DataFrame, top_n: int = 10) -> list:
        if 'Hashtags' not in df.columns:
            return []
        
        all_hashtags = []
        for tags in df['Hashtags'].dropna():
            if isinstance(tags, str):
                hashtags = re.findall(r'#(\w+)', tags)
                all_hashtags.extend(hashtags)
        
        if not all_hashtags:
            return []
        
        hashtag_counts = Counter(all_hashtags)
        return [f"#{tag}" for tag, count in hashtag_counts.most_common(top_n)]

    def _get_country_analysis(self, df: pd.DataFrame) -> dict:
        if 'Country' not in df.columns:
            return {'available': False}
        
        countries = df['Country'].dropna().unique()
        country_data = {}
        
        for country in countries:
            country_df = df[df['Country'] == country]
            sentiment_counts = country_df['Sentiment'].value_counts()
            country_data[country.strip()] = {
                'total': int(len(country_df)),
                'Positive': int(sentiment_counts.get('Positive', 0)),
                'Negative': int(sentiment_counts.get('Negative', 0)),
                'Neutral': int(sentiment_counts.get('Neutral', 0))
            }
        
        return {
            'available': True,
            'distributions': country_data
        }

    def clear_cache(self):
        self._cache = None
        self._cache_time = None


_dashboard_service = None


def get_dashboard_service() -> DashboardService:
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service


if __name__ == "__main__":
    service = DashboardService()
    data = service.get_dashboard_data()
    
    print("KPIs:", data['kpis'])
    print("Text Analysis:", data['text_analysis'])
    print("Platform Analysis:", data['platform_analysis'])
    print("Time Analysis:", data['time_analysis'])
    print("Top Keywords:", data['top_keywords'])
    print("Top Hashtags:", data['top_hashtags'])
    print("Country Analysis:", data['country_analysis'])