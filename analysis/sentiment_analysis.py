import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
# простейший анализ тональности

def sentiment_analysis(df):
    required_columns = ['processed_text', 'genre', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Отсутствуют колонки: {missing_columns}")
        return df
    
    positive_words = {
        'добрый', 'хороший', 'красивый', 'счастливый', 'радость', 'любовь', 
        'победа', 'смелый', 'умный', 'богатый', 'светлый', 'веселый', 'рад',
        'удача', 'счастье', 'любить', 'целовать', 'ласкать', 'милый', 'прекрасный',
        'сильный', 'храбрый', 'победить', 'награда', 'здоровый', 'молодой'
    }
    
    negative_words = {
        'злой', 'плохой', 'страшный', 'грустный', 'горе', 'ненависть',
        'поражение', 'трусливый', 'глупый', 'бедный', 'темный', 'печальный',
        'плакать', 'смерть', 'убить', 'бить', 'болезнь', 'старый', 'беда',
        'несчастье', 'потерять', 'погибнуть', 'зло', 'черный', 'кощей'
    }
    
    def analyze_sentiment(tokens):
        if not isinstance(tokens, list) or len(tokens) == 0:
            return 0
            
        pos_count = sum(1 for token in tokens if token in positive_words)
        neg_count = sum(1 for token in tokens if token in negative_words)
        
        total_emotional_words = pos_count + neg_count
        if total_emotional_words == 0:
            return 0
        
        sentiment_score = (pos_count - neg_count) / total_emotional_words
        return sentiment_score
    
    print("Вычисление тональности сказок...")
    df['sentiment_score'] = df['processed_text'].apply(analyze_sentiment)
    
    print(f"\nСтатистика тональности:")
    print(f"   Средняя тональность: {df['sentiment_score'].mean():.3f}")
    print(f"   Медианная тональность: {df['sentiment_score'].median():.3f}")
    print(f"   Стандартное отклонение: {df['sentiment_score'].std():.3f}")
    
    def categorize_sentiment(score):
        if score > 0.1:
            return 'положительная'
        elif score < -0.1:
            return 'отрицательная'
        else:
            return 'нейтральная'
    
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    
    plt.figure(figsize=(15, 10))
    
    # Boxplot по жанрам
    plt.subplot(2, 2, 1)
    genres = df['genre'].unique()
    sentiment_data = [df[df['genre'] == genre]['sentiment_score'].dropna() for genre in genres]
    
    box = plt.boxplot(sentiment_data, labels=genres, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'plum']
    for patch, color in zip(box['boxes'], colors[:len(genres)]):
        patch.set_facecolor(color)
    
    plt.title('Распределение тональности по жанрам', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Тональность')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    sentiment_counts = df['sentiment_category'].value_counts()
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=['lightgreen', 'lightcoral', 'lightblue'])
    plt.title('Распределение тональности всех сказок')
    
    plt.subplot(2, 2, 3)
    
    df['word_count'] = df['processed_text'].apply(len)
    
    genres = df['genre'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(genres)))
    
    for i, genre in enumerate(genres):
        genre_data = df[df['genre'] == genre]
        plt.scatter(genre_data['sentiment_score'], genre_data['word_count'], 
                   c=[colors[i]], label=genre, alpha=0.7, s=60)
    
    plt.xlabel('Тональность')
    plt.ylabel('Количество слов')
    plt.title('Тональность vs Длина сказки (в словах)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    
    # топ-5 положительных и отрицательных
    top_positive = df.nlargest(5, 'sentiment_score')[['name', 'sentiment_score']]
    top_negative = df.nsmallest(5, 'sentiment_score')[['name', 'sentiment_score']]
    
    comparison_df = pd.concat([
        top_positive.assign(type='Положительные'),
        top_negative.assign(type='Отрицательные')
    ])
    
    colors = ['lightgreen'] * 5 + ['lightcoral'] * 5
    bars = plt.barh(range(len(comparison_df)), comparison_df['sentiment_score'], color=colors)
    plt.yticks(range(len(comparison_df)), comparison_df['name'], fontsize=9)
    plt.xlabel('Тональность')
    plt.title('Топ положительных и отрицательных сказок')
    plt.grid(True, alpha=0.3, axis='x')
    
    for bar, score in zip(bars, comparison_df['sentiment_score']):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("\nСтатистика тональности по жанрам:")
    sentiment_stats = df.groupby('genre').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_category': lambda x: x.value_counts().to_dict()
    }).round(3)
    
    print(sentiment_stats)
    
    print(f"\nТоп-3 самых положительных сказок:")
    for idx, row in df.nlargest(3, 'sentiment_score')[['name', 'genre', 'sentiment_score']].iterrows():
        print(f"   {row['name']} ({row['genre']}): {row['sentiment_score']:.3f}")
    
    print(f"\nТоп-3 самых отрицательных сказок:")
    for idx, row in df.nsmallest(3, 'sentiment_score')[['name', 'genre', 'sentiment_score']].iterrows():
        print(f"   {row['name']} ({row['genre']}): {row['sentiment_score']:.3f}")
    
    return df

if __name__ == "__main__":
    try:
        df = pd.read_pickle('processed_tales_clean.pkl')
        print("Данные загружены для анализа тональности")
        df = sentiment_analysis(df)
    except Exception as e:
        print(f"Ошибка: {e}")