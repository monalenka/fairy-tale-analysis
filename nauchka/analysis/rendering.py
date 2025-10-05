import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
# облака слов по жанрам

def create_wordclouds(df):
    if 'processed_text_str' not in df.columns:
        print("Отсутствует колонка 'processed_text_str'")
        return
    
    if 'genre' not in df.columns:
        print("Отсутствует колонка 'genre'")
        return
    
    genres = df['genre'].unique()
    print(f"Создание облаков слов для {len(genres)} жанров: {list(genres)}")
    
    n_genres = len(genres)
    if n_genres == 0:
        print("Нет жанров для отображения")
        return
    
    if n_genres <= 2:
        fig, axes = plt.subplots(1, n_genres, figsize=(6 * n_genres, 5))
    elif n_genres <= 4:
        fig, axes = plt.subplots(2, (n_genres + 1) // 2, figsize=(15, 10))
    else:
        fig, axes = plt.subplots((n_genres + 2) // 3, 3, figsize=(18, 5 * ((n_genres + 2) // 3)))
    
    if n_genres == 1:
        axes = [axes]
    elif n_genres > 1 and hasattr(axes, 'flat'):
        axes = axes.flatten()
    
    color_schemes = {
        'волшебная': 'viridis',
        'о животных': 'plasma', 
        'бытовая': 'inferno',
        'героическая': 'magma',
        'шутливая': 'spring',
        'неизвестно': 'cool'
    }
    
    for idx, genre in enumerate(genres):
        if idx >= len(axes):
            break
            
        genre_text = ' '.join(df[df['genre'] == genre]['processed_text_str'].dropna())
        
        if not genre_text.strip():
            print(f"Для жанра '{genre}' нет текста")
            axes[idx].text(0.5, 0.5, f'Нет данных\nдля {genre}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
            continue
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap=color_schemes.get(genre, 'viridis'),
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=100,
            random_state=42
        ).generate(genre_text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'{genre}', fontsize=14, fontweight='bold', pad=20)
        axes[idx].axis('off')
        
        genre_df = df[df['genre'] == genre]
        word_count = len(genre_text.split())
        tale_count = len(genre_df)
        
        axes[idx].text(0.5, -0.1, f'{tale_count} сказок, {word_count} слов', 
                      transform=axes[idx].transAxes, ha='center', fontsize=10)
    
    for idx in range(len(genres), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nСтатистика по жанрам:")
    for genre in genres:
        genre_df = df[df['genre'] == genre]
        text = ' '.join(genre_df['processed_text_str'].dropna())
        words = text.split()
        unique_words = len(set(words))
        
        print(f"   {genre}: {len(genre_df)} сказок, {len(words)} слов, {unique_words} уникальных слов")

def create_overall_wordcloud(df):
    """
    Создание общего облака слов для всех сказок
    """
    if 'processed_text_str' not in df.columns:
        print("Отсутствует колонка 'processed_text_str'")
        return
    
    all_text = ' '.join(df['processed_text_str'].dropna())
    
    if not all_text.strip():
        print("Нет текста для создания облака слов")
        return
    
    print(f" Создание общего облака слов для {len(df)} сказок")
    print(f"   Всего слов: {len(all_text.split())}")
    print(f"   Уникальных слов: {len(set(all_text.split()))}")
    
    wordcloud = WordCloud(
        width=1200, 
        height=600, 
        background_color='white',
        max_words=150,
        colormap='rainbow',
        relative_scaling=0.4,
        min_font_size=8,
        max_font_size=120,
        random_state=42
    ).generate(all_text)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('ОБЛАКО СЛОВ', 
              fontsize=20, fontweight='bold', pad=20)
    plt.axis('off')
    
    stats_text = f"Всего: {len(df)} сказок, {len(all_text.split())} слов"
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        df = pd.read_pickle('processed_tales_clean.pkl')
        print("Данные загружены")
        
        create_wordclouds(df)
        
        create_overall_wordcloud(df)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")