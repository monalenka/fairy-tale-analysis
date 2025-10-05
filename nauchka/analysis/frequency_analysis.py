import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import pymorphy2

# частотный анализ, анализ по частям речи, жанрам, длине сказок
morph = pymorphy2.MorphAnalyzer()

def load_processed_data(filename):
    df = pd.read_pickle(filename)
    
    print(f"Загружено {len(df)} обработанных сказок")
    print(f"Колонки: {df.columns.tolist()}")
    print(f"Тип данных в processed_text: {type(df['processed_text'].iloc[0])}")
    
    return df

def frequency_analysis(df, top_n=30):
    all_tokens = []
    for tokens in df['processed_text']:
        all_tokens.extend(tokens)
    
    freq_dist = Counter(all_tokens)
    top_words = freq_dist.most_common(top_n)
    
    words, counts = zip(*top_words)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(words)), counts, color='skyblue')
    plt.title(f'Топ-{top_n} самых частотных слов в русских народных сказках', fontsize=14)
    plt.xlabel('Слова')
    plt.ylabel('Частота')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nТоп-{top_n} самых частых слов:")
    for i, (word, count) in enumerate(top_words, 1):
        print(f"{i:2d}. {word:<15} - {count:>4} раз")
    
    return freq_dist

def pos_analysis(df):
    pos_counter = Counter()
    word_count = 0
    
    for tokens in df['processed_text']:
        for token in tokens:
            try:
                parsed = morph.parse(token)[0]
                pos = parsed.tag.POS
                if pos:
                    pos_counter[pos] += 1
                    word_count += 1
            except Exception as e:
                continue
    
    pos_translation = {
        'NOUN': 'Существительные',
        'VERB': 'Глаголы', 
        'ADJF': 'Прилагательные',
        'ADJS': 'Краткие прилагательные',
        'ADVB': 'Наречия',
        'PRTF': 'Причастия',
        'PRTS': 'Деепричастия',
        'INFN': 'Инфинитивы',
        'NUMR': 'Числительные',
        'CONJ': 'Союзы',
        'PREP': 'Предлоги',
        'NPRO': 'Местоимения'
    }
    
    top_pos = pos_counter.most_common(10)
    pos_names = [pos_translation.get(pos, pos) for pos, count in top_pos]
    pos_counts = [count for pos, count in top_pos]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(range(len(pos_counts)))
    wedges, texts, autotexts = plt.pie(pos_counts, labels=pos_names, autopct='%1.1f%%', 
                                      startangle=90, colors=colors)
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    plt.title('Распределение частей речи в сказках', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    print("\nРаспределение частей речи:")
    total_words = sum(pos_counts)
    for (pos, count), name in zip(top_pos, pos_names):
        percent = (count / total_words) * 100
        print(f"{name:<20} - {count:>5} слов ({percent:.1f}%)")
    
    print(f"\nВсего проанализировано слов: {total_words}")
    
    return pos_counter

def genre_analysis(df):
    print("\n" + "="*50)
    print("АНАЛИЗ ПО ЖАНРАМ")
    print("="*50)
    
    for genre in df['genre'].unique():
        genre_df = df[df['genre'] == genre]
        all_tokens = []
        for tokens in genre_df['processed_text']:
            all_tokens.extend(tokens)
        
        if all_tokens:
            freq_dist = Counter(all_tokens)
            top_words = freq_dist.most_common(8)
            
            print(f"\n {genre.upper()} ({len(genre_df)} сказок):")
            for i, (word, count) in enumerate(top_words, 1):
                print(f"   {i}. {word:<12} - {count:>3} раз")
            
            print(f"   Всего уникальных слов: {len(freq_dist)}")

def length_analysis(df):
    print("\n" + "="*50)
    print("АНАЛИЗ ДЛИНЫ СКАЗОК")
    print("="*50)
    
    df['word_count'] = df['processed_text'].apply(len)
    
    print(f"Общее количество слов: {df['word_count'].sum()}")
    print(f"Средняя длина сказки: {df['word_count'].mean():.1f} слов")
    print(f"Минимальная длина: {df['word_count'].min()} слов")
    print(f"Максимальная длина: {df['word_count'].max()} слов")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['word_count'], bins=20, color='lightgreen', edgecolor='black')
    plt.title('Распределение длины сказок')
    plt.xlabel('Количество слов')
    plt.ylabel('Количество сказок')
    
    plt.subplot(1, 2, 2)
    df.boxplot(column='word_count')
    plt.title('Boxplot длины сказок')
    plt.ylabel('Количество слов')
    
    plt.tight_layout()
    plt.show()
    
    length_stats = df.groupby('length')['word_count'].agg(['count', 'mean', 'min', 'max'])
    print("\nСтатистика по категориям длины:")
    print(length_stats.round(1))

if __name__ == "__main__":
    df = load_processed_data('processed_tales_clean.pkl')
    
    if 'processed_text' not in df.columns:
        print("Ошибка: колонка 'processed_text' не найдена!")
    else:
        print("Начинаем анализ сказок...")
        
        freq_dist = frequency_analysis(df)
        
        pos_dist = pos_analysis(df)
        
        genre_analysis(df)
        
        length_analysis(df)
        
        print("\nВсе анализы завершены корректно.")