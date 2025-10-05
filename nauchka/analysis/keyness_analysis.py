import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# ключевые слова в сказках

def keyness_analysis(df):
    if 'genre' not in df.columns or 'processed_text_str' not in df.columns:
        print("Ошибка: отсутствуют необходимые колонки 'genre' или 'processed_text_str'")
        return None, None, None
    
    genres = df['genre'].unique()
    print(f"Найдены жанры: {list(genres)}")
    
    genre_texts = {}
    for genre in genres:
        genre_df = df[df['genre'] == genre]
        if len(genre_df) > 0:
            genre_texts[genre] = ' '.join(genre_df['processed_text_str'].tolist())
        else:
            genre_texts[genre] = ""
    
    vectorizer = TfidfVectorizer(max_features=20, min_df=2)
    try:
        tfidf_matrix = vectorizer.fit_transform(list(genre_texts.values()))
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        print(f"Ошибка при создании TF-IDF матрицы: {e}")
        return None, None, None
    
    n_genres = len(genres)
    if n_genres == 0:
        print("Нет жанров для анализа")
        return None, None, None
        
    fig, axes = plt.subplots(n_genres, 1, figsize=(12, 4*n_genres))
    if n_genres == 1:
        axes = [axes]
    
    for idx, genre in enumerate(genres):
        if idx >= len(axes):
            break
            
        tfidf_scores = tfidf_matrix[idx].toarray().flatten()
        
        # топ-10 слов с наибольшим TF-IDF
        top_indices = tfidf_scores.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [tfidf_scores[i] for i in top_indices]
        
        sorted_indices = np.argsort(top_scores)[::-1]
        top_words = [top_words[i] for i in sorted_indices]
        top_scores = [top_scores[i] for i in sorted_indices]
        
        axes[idx].barh(top_words, top_scores, color='lightcoral')
        axes[idx].set_title(f'Ключевые слова для жанра: {genre}', fontsize=14)
        axes[idx].set_xlabel('TF-IDF Score')
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
    for idx, genre in enumerate(genres):
        tfidf_scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = tfidf_scores.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [tfidf_scores[i] for i in top_indices]
        
        # по убыванию TF-IDF
        sorted_pairs = sorted(zip(top_words, top_scores), key=lambda x: x[1], reverse=True)
        
        print(f"\n {genre.upper()}:")
        for i, (word, score) in enumerate(sorted_pairs, 1):
            print(f"   {i:2d}. {word:<15} - TF-IDF: {score:.4f}")
    
    return genre_texts, tfidf_matrix, feature_names

def compare_genres(df, genre1, genre2):
    """
    Сравнительный анализ двух жанров для нахождения характерных слов
    """
    if genre1 not in df['genre'].values or genre2 not in df['genre'].values:
        print(f"Ошибка: один из жанров ({genre1} или {genre2}) не найден в данных")
        return
    
    print(f"\n" + "="*50)
    print(f"СРАВНЕНИЕ ЖАНРОВ: {genre1.upper()} vs {genre2.upper()}")
    print("="*50)
    
    genre1_words = []
    for text in df[df['genre'] == genre1]['processed_text']:
        genre1_words.extend(text)
    
    genre2_words = []
    for text in df[df['genre'] == genre2]['processed_text']:
        genre2_words.extend(text)
    
    genre1_freq = Counter(genre1_words)
    genre2_freq = Counter(genre2_words)
    
    all_words = set(genre1_freq.keys()).union(set(genre2_freq.keys()))
    
    genre1_specific = {}
    genre2_specific = {}
    
    for word in all_words:
        freq1 = genre1_freq.get(word, 0)
        freq2 = genre2_freq.get(word, 0)
        
        if freq1 + freq2 < 5:
            continue
            
        total1 = len(genre1_words)
        total2 = len(genre2_words)
        
        if total1 > 0 and total2 > 0:
            rel_freq1 = (freq1 / total1) * 1000  # на 1000 слов
            rel_freq2 = (freq2 / total2) * 1000
            
            # log-ratio (мера ключевости)
            if rel_freq1 > 0 and rel_freq2 > 0:
                log_ratio = np.log2(rel_freq1 / rel_freq2)
            elif rel_freq1 > 0:
                log_ratio = 10
            elif rel_freq2 > 0:
                log_ratio = -10
            else:
                continue
            
            if log_ratio > 1.5 and freq1 >= 3:
                genre1_specific[word] = (freq1, rel_freq1, log_ratio)
            elif log_ratio < -1.5 and freq2 >= 3:
                genre2_specific[word] = (freq2, rel_freq2, abs(log_ratio))
    
    print(f"\nСлова, характерные для '{genre1}':")
    sorted_genre1 = sorted(genre1_specific.items(), key=lambda x: x[1][2], reverse=True)[:15]
    for word, (freq, rel_freq, log_ratio) in sorted_genre1:
        print(f"   {word:<15} - частота: {freq:>3}, относ.частота: {rel_freq:>5.1f}, keyness: {log_ratio:>4.2f}")
    
    print(f"\nСлова, характерные для '{genre2}':")
    sorted_genre2 = sorted(genre2_specific.items(), key=lambda x: x[1][2], reverse=True)[:15]
    for word, (freq, rel_freq, log_ratio) in sorted_genre2:
        print(f"   {word:<15} - частота: {freq:>3}, относ.частота: {rel_freq:>5.1f}, keyness: {log_ratio:>4.2f}")

if __name__ == "__main__":
    try:
        df = pd.read_pickle('processed_tales_clean.pkl')
        print("Данные успешно загружены из processed_tales_clean.pkl")
        print(f"Размер данных: {len(df)} сказок")
        print(f"Жанры: {df['genre'].unique()}")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        exit()
    
    print("\nЗапуск Keyness-анализа.")
    genre_texts, tfidf_matrix, feature_names = keyness_analysis(df)
    
    if genre_texts is not None and len(df['genre'].unique()) >= 2:
        genres = df['genre'].unique()[:2]
        print(f"\nСравниваю жанры: {genres[0]} и {genres[1]}")
        compare_genres(df, genres[0], genres[1])