import pandas as pd
import numpy as np
import json

df = pd.read_pickle('processed_tales.pkl')

print("Поиск сказок с проблемными жанрами...")

print(f"Всего сказок: {len(df)}")

nan_genres = df[df['genre'].isna()]
print(f"\nСказки с NaN в жанре: {len(nan_genres)}")
if len(nan_genres) > 0:
    print("Индексы сказок с NaN:")
    for idx in nan_genres.index:
        print(f"  Индекс {idx}: {df.loc[idx, 'name']}")

float_genres = df[df['genre'].apply(lambda x: isinstance(x, float) and not pd.isna(x))]
print(f"\nСказки с float в жанре: {len(float_genres)}")
if len(float_genres) > 0:
    print("Индексы сказок с float:")
    for idx in float_genres.index:
        print(f"  Индекс {idx}: {df.loc[idx, 'name']} - жанр: {df.loc[idx, 'genre']}")

empty_genres = df[df['genre'].apply(lambda x: isinstance(x, str) and x.strip() == '')]
print(f"\nСказки с пустыми жанрами: {len(empty_genres)}")
if len(empty_genres) > 0:
    print("Индексы сказок с пустыми жанрами:")
    for idx in empty_genres.index:
        print(f"  Индекс {idx}: {df.loc[idx, 'name']}")

print(f"\nВсе уникальные значения жанра:")
for genre in df['genre'].unique():
    print(f"  Тип: {type(genre)}, Значение: '{genre}'")

def clean_and_save_multiformat(df, base_filename='processed_tales_clean'):
    df_clean = df.copy()
    
    df_clean = df_clean[~df_clean['genre'].isna()]
    
    df_clean['genre'] = df_clean['genre'].astype(str).str.strip()
    
    df_clean = df_clean[df_clean['genre'] != '']
    
    df_clean = df_clean[~df_clean['genre'].str.match(r'^\d+\.?\d*$')]
    
    print(f"\nПосле очистки: {len(df_clean)} сказок из {len(df)}")
    print(f"Оставшиеся жанры: {df_clean['genre'].unique()}")
    
    pickle_filename = f'{base_filename}.pkl'
    df_clean.to_pickle(pickle_filename)
    print(f"Pickle: сохранено в '{pickle_filename}'")
    
    csv_filename = f'{base_filename}.csv'
    
    df_csv = df_clean.copy()
    
    if 'processed_text' in df_csv.columns:
        df_csv['processed_text'] = df_csv['processed_text'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
    
    if 'characters' in df_csv.columns:
        df_csv['characters'] = df_csv['characters'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
    
    df_csv.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"CSV: сохранено в '{csv_filename}'")
    
    json_filename = f'{base_filename}.json'
    
    json_data = df_clean.to_dict('records')
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"JSON: сохранено в '{json_filename}'")
    
    jsonl_filename = f'{base_filename}.jsonl'
    
    with open(jsonl_filename, 'w', encoding='utf-8') as f:
        for record in json_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"JSONL: сохранено в '{jsonl_filename}'")
    
    print(f"\n📁 Созданные файлы:")
    print(f"   - {pickle_filename} (полные данные, для анализа в Python)")
    print(f"   - {csv_filename} (табличный формат, для Excel)")
    print(f"   - {json_filename} (структурированные данные, для веб-приложений)")
    print(f"   - {jsonl_filename} (строчный JSON, для обработки больших данных)")
    
    return df_clean

df_clean = clean_and_save_multiformat(df)

print("\nПроверка очищенных данных:")
print(f"Типы данных в колонке жанр: {df_clean['genre'].apply(type).unique()}")
print(f"Есть ли NaN: {df_clean['genre'].isna().any()}")