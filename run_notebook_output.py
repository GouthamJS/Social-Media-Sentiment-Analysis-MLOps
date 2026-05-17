"""
Execute all notebook cells and display outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re
import os
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

NLTK_AVAILABLE = False
WORDCLOUD_AVAILABLE = False

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except:
    pass

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except:
    pass

OUTPUT_DIR = 'artifacts/eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("SENTIMENT ANALYSIS NOTEBOOK - ALL CELL OUTPUTS")
print("="*60)

# ============================================================================
# CELL 1: Import Libraries
# ============================================================================
print("\n" + "="*60)
print("CELL 1: Import Libraries")
print("="*60)
print("Libraries imported successfully!")
print(f"NLTK available: {NLTK_AVAILABLE}")
print(f"WordCloud available: {WORDCLOUD_AVAILABLE}")

# ============================================================================
# CELL 2: Data Loading
# ============================================================================
print("\n" + "="*60)
print("CELL 2: Data Loading")
print("="*60)

data_path = 'artificats/raw/sentimentdataset.csv'
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# CELL 3: Basic Statistics
# ============================================================================
print("\n" + "="*60)
print("CELL 3: Basic Statistics")
print("="*60)
print("\nDataset Info:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSentiment Value Counts (Original):")
print(df['Sentiment'].value_counts())

# ============================================================================
# CELL 4: Sentiment Distribution
# ============================================================================
print("\n" + "="*60)
print("CELL 4: Sentiment Distribution Visualization")
print("="*60)

plt.figure(figsize=(10, 5))
df['Sentiment'].value_counts().plot(kind='bar', color=['#43e97b', '#f5576c', '#4facfe', '#ff9a9e', '#a18cd1'])
plt.title('Sentiment Distribution (Original)', fontsize=14)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/sentiment_distribution.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/sentiment_distribution.png")

# ============================================================================
# CELL 5: Text Length Analysis
# ============================================================================
print("\n" + "="*60)
print("CELL 5: Text Length Analysis")
print("="*60)

df['text_length'] = df['Text'].apply(lambda x: len(str(x)))
df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))

print("Text Length Statistics:")
print(f"Average: {df['text_length'].mean():.2f}")
print(f"Max: {df['text_length'].max()}")
print(f"Min: {df['text_length'].min()}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['text_length'], bins=30, color='steelblue', edgecolor='black')
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['word_count'], bins=20, color='coral', edgecolor='black')
plt.title('Word Count Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/text_length_analysis.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/text_length_analysis.png")

# ============================================================================
# CELL 6: Platform vs Sentiment Analysis
# ============================================================================
print("\n" + "="*60)
print("CELL 6: Platform vs Sentiment Analysis")
print("="*60)

if 'Platform' in df.columns:
    plt.figure(figsize=(12, 6))
    platform_sentiment = pd.crosstab(df['Platform'].str.strip(), df['Sentiment'])
    platform_sentiment.plot(kind='bar', stacked=True)
    plt.title('Platform vs Sentiment Distribution')
    plt.xlabel('Platform')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/platform_sentiment.png', dpi=150)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/platform_sentiment.png")
    print("\nPlatform Distribution:")
    print(df['Platform'].value_counts())

# ============================================================================
# CELL 7: Sentiment Trend Over Time
# ============================================================================
print("\n" + "="*60)
print("CELL 7: Sentiment Trend Over Time")
print("="*60)

if 'Year' in df.columns:
    plt.figure(figsize=(12, 6))
    yearly_trend = df.groupby(['Year', 'Sentiment']).size().unstack(fill_value=0)
    yearly_trend.plot(kind='line', marker='o')
    plt.title('Sentiment Trend Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sentiment_trend_yearly.png', dpi=150)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/sentiment_trend_yearly.png")
    print("\nYearly Distribution:")
    print(yearly_trend)

# ============================================================================
# CELL 8: Country Analysis
# ============================================================================
print("\n" + "="*60)
print("CELL 8: Country Analysis")
print("="*60)

if 'Country' in df.columns:
    plt.figure(figsize=(12, 6))
    country_sentiment = pd.crosstab(df['Country'].str.strip(), df['Sentiment'])
    top_countries = country_sentiment.sum(axis=1).nlargest(10)
    country_sentiment.loc[top_countries.index].plot(kind='bar', stacked=True)
    plt.title('Sentiment by Country (Top 10)')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/country_sentiment.png', dpi=150)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/country_sentiment.png")

# ============================================================================
# CELL 9: Top Keywords
# ============================================================================
print("\n" + "="*60)
print("CELL 9: Top Keywords Analysis")
print("="*60)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['Text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['clean_text'].fillna(''))
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
keyword_scores = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 6))
keywords = [k for k, s in keyword_scores[:15]]
scores = [s for k, s in keyword_scores[:15]]
plt.barh(range(len(keywords)), scores, color='steelblue')
plt.yticks(range(len(keywords)), keywords)
plt.xlabel('TF-IDF Score')
plt.title('Top 15 Keywords')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/top_keywords.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/top_keywords.png")

# ============================================================================
# CELL 10: Top Hashtags
# ============================================================================
print("\n" + "="*60)
print("CELL 10: Top Hashtags Analysis")
print("="*60)

if 'Hashtags' in df.columns:
    all_hashtags = []
    for tags in df['Hashtags'].dropna():
        if isinstance(tags, str):
            hashtags = re.findall(r'#(\w+)', tags)
            all_hashtags.extend(hashtags)
    
    hashtag_counts = Counter(all_hashtags).most_common(15)
    
    plt.figure(figsize=(10, 6))
    hashtags = [f"#{h}" for h, c in hashtag_counts]
    count_vals = [c for h, c in hashtag_counts]
    plt.barh(range(len(hashtags)), count_vals, color='coral')
    plt.yticks(range(len(hashtags)), hashtags)
    plt.xlabel('Count')
    plt.title('Top 15 Hashtags')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_hashtags.png', dpi=150)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/top_hashtags.png")

# ============================================================================
# CELL 11: Word Cloud Analysis by Sentiment
# ============================================================================
print("\n" + "="*60)
print("CELL 11: Word Cloud Analysis by Sentiment")
print("="*60)

if WORDCLOUD_AVAILABLE:
    sentiments = df['Sentiment'].unique()
    color_maps = {'Positive': 'Greens', 'Negative': 'Reds', 'Neutral': 'Blues'}
    default_colors = ['Greens', 'Reds', 'Blues', 'Purples', 'Oranges']
    
    for i, sentiment in enumerate(sentiments[:3]):
        sentiment_texts = df[df['Sentiment'] == sentiment]['clean_text']
        color = color_maps.get(sentiment, default_colors[i % len(default_colors)])
        text = ' '.join(sentiment_texts.dropna().astype(str))
        if len(text) > 0:
            wc = WordCloud(width=800, height=400, background_color='white', 
                         colormap=color, max_words=100).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud - {sentiment}')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/wordcloud_{sentiment.lower()}.png', dpi=150)
            plt.show()
            print(f"Saved: {OUTPUT_DIR}/wordcloud_{sentiment.lower()}.png")
else:
    print("WordCloud library not installed. Skipping word cloud generation.")

# ============================================================================
# CELL 12: N-gram Analysis - Unigrams
# ============================================================================
print("\n" + "="*60)
print("CELL 12: N-gram Analysis - Unigrams (1-gram)")
print("="*60)

def get_ngrams(text, n):
    words = str(text).lower().split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

all_unigrams = []
for text in df['Text']:
    all_unigrams.extend(get_ngrams(text, 1))

unigram_counts = Counter(all_unigrams).most_common(15)

plt.figure(figsize=(10, 6))
unigrams = [n for n, c in unigram_counts]
counts = [c for n, c in unigram_counts]
plt.barh(range(len(unigrams)), counts, color='steelblue')
plt.yticks(range(len(unigrams)), unigrams)
plt.xlabel('Count')
plt.title('Top 15 Unigrams (1-gram)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1gram_analysis.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/1gram_analysis.png")
print("\nTop 15 Unigrams:")
for i, (ngram, count) in enumerate(unigram_counts, 1):
    print(f"  {i}. '{ngram}': {count}")

# ============================================================================
# CELL 13: N-gram Analysis - Bigrams
# ============================================================================
print("\n" + "="*60)
print("CELL 13: N-gram Analysis - Bigrams (2-gram)")
print("="*60)

all_bigrams = []
for text in df['Text']:
    all_bigrams.extend(get_ngrams(text, 2))

bigram_counts = Counter(all_bigrams).most_common(15)

plt.figure(figsize=(10, 6))
bigrams_list = [n for n, c in bigram_counts]
counts = [c for n, c in bigram_counts]
plt.barh(range(len(bigrams_list)), counts, color='coral')
plt.yticks(range(len(bigrams_list)), bigrams_list)
plt.xlabel('Count')
plt.title('Top 15 Bigrams (2-gram)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2gram_analysis.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/2gram_analysis.png")
print("\nTop 15 Bigrams:")
for i, (ngram, count) in enumerate(bigram_counts, 1):
    print(f"  {i}. '{ngram}': {count}")

# ============================================================================
# CELL 14: N-gram Analysis - Trigrams
# ============================================================================
print("\n" + "="*60)
print("CELL 14: N-gram Analysis - Trigrams (3-gram)")
print("="*60)

all_trigrams = []
for text in df['Text']:
    all_trigrams.extend(get_ngrams(text, 3))

trigram_counts = Counter(all_trigrams).most_common(15)

plt.figure(figsize=(10, 6))
trigrams_list = [n for n, c in trigram_counts]
counts = [c for n, c in trigram_counts]
plt.barh(range(len(trigrams_list)), counts, color='green')
plt.yticks(range(len(trigrams_list)), trigrams_list)
plt.xlabel('Count')
plt.title('Top 15 Trigrams (3-gram)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3gram_analysis.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/3gram_analysis.png")
print("\nTop 15 Trigrams:")
for i, (ngram, count) in enumerate(trigram_counts, 1):
    print(f"  {i}. '{ngram}': {count}")

# ============================================================================
# CELL 15: N-gram Analysis by Sentiment
# ============================================================================
print("\n" + "="*60)
print("CELL 15: N-gram Analysis by Sentiment")
print("="*60)

plt.figure(figsize=(14, 10))
sentiments = df['Sentiment'].unique()

for idx, sentiment in enumerate(sentiments[:3], 1):
    sentiment_df = df[df['Sentiment'] == sentiment]
    all_bigrams_sent = []
    for text in sentiment_df['Text']:
        all_bigrams_sent.extend(get_ngrams(text, 2))
    
    bigram_counts_sent = Counter(all_bigrams_sent).most_common(8)
    
    plt.subplot(2, 3, idx)
    if bigram_counts_sent:
        ngrams, counts = zip(*bigram_counts_sent)
        plt.barh(range(len(ngrams)), counts, color=plt.cm.Set2(idx))
        plt.yticks(range(len(ngrams)), ngrams)
        plt.title(f'{sentiment} - Top Bigrams')
        plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ngram_by_sentiment.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/ngram_by_sentiment.png")

# ============================================================================
# CELL 16: Label Mapping
# ============================================================================
print("\n" + "="*60)
print("CELL 16: Label Mapping to Positive/Negative/Neutral")
print("="*60)

positive_labels = ['Positive', 'Happiness', 'Joy', 'Love', 'Amusement', 'Enjoyment', 'Admiration', 
               'Affection', 'Awe', 'Surprise', 'Adoration', 'Excitement', 'Anticipation',
               'Calmness', 'Contentment', 'Serenity', 'Gratitude', 'Hope', 'Empowerment',
               'Compassion', 'Tenderness', 'Enthusiasm', 'Fulfillment', 'Elation', 
               'Euphoria', 'Pride', 'Kind', 'Acceptance', 'Curiosity', 'Determination']

negative_labels = ['Negative', 'Anger', 'Fear', 'Sadness', 'Disgust', 'Disappointed',
                 'Bitter', 'Confusion', 'Frustration', 'Shame', 'Jealousy', 'Resentment',
                 'Boredom', 'Anxiety', 'Intimidation', 'Helplessness', 'Envy', 'Regret',
                 'Despair', 'Grief', 'Loneliness', 'Numbness', 'Melancholy', 'Ambivalence',
                 'Indifference', 'Nostalgia']

neutral_labels = ['Neutral', 'Confusion', 'Curiosity', 'Indifference', 'Numbness',
               'Contentment', 'Acceptance']

def map_sentiment(label):
    label = str(label).strip()
    if label in positive_labels:
        return 'positive'
    elif label in negative_labels:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_mapped'] = df['Sentiment'].apply(map_sentiment)

print("Mapped Sentiment Distribution:")
print(df['sentiment_mapped'].value_counts())

plt.figure(figsize=(8, 5))
colors = {'positive': '#43e97b', 'negative': '#f5576c', 'neutral': '#4facfe'}
df['sentiment_mapped'].value_counts().plot(kind='bar', color=[colors.get(x, 'gray') for x in df['sentiment_mapped'].value_counts().index])
plt.title('Mapped Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/sentiment_mapped_distribution.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/sentiment_mapped_distribution.png")

# ============================================================================
# CELL 17: BERT Embeddings Setup
# ============================================================================
print("\n" + "="*60)
print("CELL 17: BERT Embeddings Setup")
print("="*60)

BERT_AVAILABLE = False
try:
    from transformers import BertTokenizer, BertModel
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()
    
    print("BERT model loaded successfully!")
    BERT_AVAILABLE = True
except Exception as e:
    print(f"BERT model not available: {e}")
    print("Using TF-IDF as alternative feature engineering...")

# ============================================================================
# CELL 18: Feature Engineering
# ============================================================================
print("\n" + "="*60)
print("CELL 18: Feature Engineering")
print("="*60)

if BERT_AVAILABLE:
    def get_bert_embeddings(texts, tokenizer, model, device, batch_size=32, max_length=128):
        embeddings = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)
                if (i + batch_size) % 100 == 0:
                    print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        return np.array(embeddings)
    
    texts = df['Text'].tolist()
    print(f"Generating BERT embeddings for {len(texts)} texts...")
    embeddings = get_bert_embeddings(texts, tokenizer, model, device, batch_size=32)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    np.save('bert_embeddings.npy', embeddings)
    print("BERT embeddings saved to 'bert_embeddings.npy'")
else:
    print("Using TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=768)
    embeddings = tfidf.fit_transform(df['clean_text'].fillna('')).toarray()
    print(f"TF-IDF embeddings shape: {embeddings.shape}")
    np.save('tfidf_embeddings.npy', embeddings)
    print("TF-IDF embeddings saved to 'tfidf_embeddings.npy'")

# ============================================================================
# CELL 19: Model Training
# ============================================================================
print("\n" + "="*60)
print("CELL 19: Model Training - Multinomial Logistic Regression")
print("="*60)

X = embeddings
y = np.array(df['sentiment_mapped'].tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"\nTraining set distribution:")
print(pd.Series(y_train).value_counts())

print("\nTraining Multinomial Logistic Regression...")
lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# CELL 20: Model Evaluation
# ============================================================================
print("\n" + "="*60)
print("CELL 20: Model Evaluation")
print("="*60)

y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['positive', 'neutral', 'negative'],
            yticklabels=['positive', 'neutral', 'negative'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/confusion_matrix.png")

# Predicted Distribution
plt.figure(figsize=(8, 5))
pd.Series(y_pred).value_counts().plot(kind='bar', color=['#43e97b', '#4facfe', '#f5576c'])
plt.title('Predicted Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/predicted_distribution.png', dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR}/predicted_distribution.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("PIPELINE COMPLETE - SUMMARY")
print("="*60)
print("\nAll EDA visualizations saved to:", OUTPUT_DIR)
print("\nFiles created:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")