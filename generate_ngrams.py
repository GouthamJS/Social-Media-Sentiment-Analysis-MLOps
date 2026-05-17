import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import os

OUTPUT_DIR = 'artifacts/eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv('artificats/raw/sentimentdataset.csv')
df.columns = df.columns.str.strip()

def get_ngrams(text, n):
    words = str(text).lower().split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def plot_ngrams(n, top_n=15, title_prefix='Top'):
    all_ngrams = []
    for text in df['Text']:
        all_ngrams.extend(get_ngrams(text, n))

    ngram_counts = Counter(all_ngrams).most_common(top_n)
    ngrams, counts = zip(*ngram_counts)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(ngrams)), counts, color='steelblue')
    plt.yticks(range(len(ngrams)), ngrams)
    plt.xlabel('Count')
    plt.title(f'{title_prefix} {n}-grams (Top {top_n})')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    filename = f'{n}gram_analysis.png'
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150)
    plt.close()
    print(f'Saved: {OUTPUT_DIR}/{filename}')

    return ngram_counts

print('Generating N-gram analysis images...')

unigrams = plot_ngrams(1, 15, 'Unigram')

bigrams = plot_ngrams(2, 15, 'Bigram')

trigrams = plot_ngrams(3, 15, 'Trigram')

print('Generating ngram_by_sentiment.png...')
plt.figure(figsize=(14, 10))
sentiments = df['Sentiment'].unique()

for idx, sentiment in enumerate(sentiments[:3], 1):
    sentiment_df = df[df['Sentiment'] == sentiment]
    all_bigrams = []
    for text in sentiment_df['Text']:
        all_bigrams.extend(get_ngrams(text, 2))

    bigram_counts = Counter(all_bigrams).most_common(8)

    plt.subplot(2, 3, idx)
    if bigram_counts:
        ngrams, counts = zip(*bigram_counts)
        plt.barh(range(len(ngrams)), counts, color=plt.cm.Set2(idx))
        plt.yticks(range(len(ngrams)), ngrams)
        plt.title(f'{sentiment} - Top Bigrams')
        plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ngram_by_sentiment.png', dpi=150)
plt.close()
print(f'Saved: {OUTPUT_DIR}/ngram_by_sentiment.png')

print('\nAll N-gram analysis images generated!')