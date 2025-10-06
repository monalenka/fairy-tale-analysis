import pandas as pd
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
from wordcloud import WordCloud
import nltk
#pip install seaborn, nltk, wordcloud, pymorphy2, sklearn
#python preprocessing.py
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# подготовка сказок
morph = pymorphy2.MorphAnalyzer()

def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Ошибка в строке {i}: {e}")
    return pd.DataFrame(data)

def get_stopwords():
    """список стоп-слов"""
    stop_words = set(stopwords.words('russian'))
    fairy_tale_stopwords = {
        'мы', 'свой', 'ты', 'сам', 'мой',
        'тут', 'вот', 'это', 'как', 'так',
        'очень', 'много', 'мало', 'еще', 'уже', 'опять', 'опятьтаки', 'ведь'
    }
    return stop_words.union(fairy_tale_stopwords)

def preprocess_text(text, stop_words):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    text = re.sub(r'[^а-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        tokens = word_tokenize(text, language='russian')
    except:
        tokens = text.split()
    
    lemmas = []
    for token in tokens:
        if len(token) > 2:
            try:
                lemma = morph.parse(token)[0].normal_form
                if lemma not in stop_words:
                    lemmas.append(lemma)
            except:
                if token not in stop_words:
                    lemmas.append(token)
    
    return lemmas

def process_data(df):
    stop_words = get_stopwords()
    
    print("Начата обработка текстов...")
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, stop_words))
    df['processed_text_str'] = df['processed_text'].apply(' '.join)
    
    initial_count = len(df)
    df = df[df['processed_text_str'].str.len() > 0]
    final_count = len(df)
    
    print(f"Обработано {final_count} из {initial_count} сказок")
    
    return df

if __name__ == "__main__":
    try:
        df = load_data('good_tales.jsonl')
        print(f"Загружено {len(df)} сказок")
        print(f"Жанры в датасете: {df['genre'].unique()}")
        
        df = process_data(df)
        
        print("\nСтатистика:")
        print(f"Общее количество слов: {df['processed_text'].apply(len).sum()}")
        print(f"Средняя длина сказки: {df['processed_text'].apply(len).mean():.1f} слов")

        df.to_csv('processed_tales_clean.csv', index=False, encoding='utf-8')
        df.to_pickle('processed_tales_clean.pkl')
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")