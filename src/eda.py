import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class EDA:
    def __init__(self, df: pd.DataFrame, output_dir: str = None):
        self.df = df
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'artifacts', 'eda')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')

    def generate_all(self):
        self.plot_sentiment_distribution()
        self.plot_text_length_analysis()
        if 'Platform' in self.df.columns:
            self.plot_platform_sentiment()
        print(f"EDA plots saved to: {self.output_dir}")

    def plot_sentiment_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sentiment_counts = self.df['Sentiment'].value_counts()
        colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
        bar_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]
        
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, edgecolor='white')
        
        for bar, count in zip(bars, sentiment_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   f'{count}\n({count/len(self.df)*100:.1f}%)', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(sentiment_counts.values) * 1.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_distribution.png'), dpi=150)
        plt.close()
        print(f"  - Saved: sentiment_distribution.png")

    def plot_text_length_analysis(self):
        if 'Text' in self.df.columns:
            text_col = 'Text'
        elif 'clean_text' in self.df.columns:
            text_col = 'clean_text'
        else:
            text_col = self.df.columns[0]
        
        self.df['text_length'] = self.df[text_col].astype(str).str.len()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(self.df['text_length'], bins=30, color='#3498db', edgecolor='white', alpha=0.8)
        axes[0].axvline(self.df['text_length'].mean(), color='#e74c3c', linestyle='--', 
                        linewidth=2, label=f'Mean: {self.df["text_length"].mean():.1f}')
        axes[0].axvline(self.df['text_length'].median(), color='#2ecc71', linestyle='--', 
                        linewidth=2, label=f'Median: {self.df["text_length"].median():.1f}')
        axes[0].set_xlabel('Text Length (characters)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Text Length Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        sentiment_lengths = self.df.groupby('Sentiment')['text_length'].mean().sort_values(ascending=False)
        colors = [colors_map.get(s, '#95a5a6') for s in sentiment_lengths.index]
        axes[1].bar(sentiment_lengths.index, sentiment_lengths.values, color=colors, edgecolor='white')
        axes[1].set_xlabel('Sentiment', fontsize=11)
        axes[1].set_ylabel('Average Text Length', fontsize=11)
        axes[1].set_title('Average Text Length by Sentiment', fontsize=12, fontweight='bold')
        
        for i, v in enumerate(sentiment_lengths.values):
            axes[1].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'text_length_analysis.png'), dpi=150)
        plt.close()
        print(f"  - Saved: text_length_analysis.png")

    def plot_platform_sentiment(self):
        if 'Platform' not in self.df.columns:
            print("  - Skipped: Platform column not found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        platform_sentiment = pd.crosstab(self.df['Platform'], self.df['Sentiment'])
        platform_sentiment.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c', '#3498db'], 
                                edgecolor='white', width=0.7)
        axes[0].set_xlabel('Platform', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Sentiment Distribution by Platform', fontsize=12, fontweight='bold')
        axes[0].legend(title='Sentiment')
        axes[0].tick_params(axis='x', rotation=0)
        
        platform_pct = platform_sentiment.div(platform_sentiment.sum(axis=1), axis=0) * 100
        platform_pct.plot(kind='bar', stacked=True, ax=axes[1], 
                         color=['#2ecc71', '#e74c3c', '#3498db'], edgecolor='white', width=0.7)
        axes[1].set_xlabel('Platform', fontsize=11)
        axes[1].set_ylabel('Percentage (%)', fontsize=11)
        axes[1].set_title('Sentiment Percentage by Platform', fontsize=12, fontweight='bold')
        axes[1].legend(title='Sentiment')
        axes[1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'platform_sentiment.png'), dpi=150)
        plt.close()
        print(f"  - Saved: platform_sentiment.png")

    def get_summary_stats(self) -> dict:
        stats = {
            'total_records': len(self.df),
            'sentiment_distribution': self.df['Sentiment'].value_counts().to_dict(),
            'platform_distribution': self.df['Platform'].value_counts().to_dict() if 'Platform' in self.df.columns else {},
            'avg_text_length': float(self.df['text_length'].mean()) if 'text_length' in self.df.columns else 0
        }
        return stats


colors_map = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}


if __name__ == "__main__":
    from data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    eda = EDA(df)
    eda.generate_all()
    print("\nSummary Stats:", eda.get_summary_stats())