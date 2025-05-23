import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime, timedelta

def get_bear_cluster_centers(bears_df, n_clusters=3):
    # Извлекаем год из даты
    bears_df['timestamp'] = pd.to_datetime(bears_df['timestamp'])
    bears_df['year'] = bears_df['timestamp'].dt.year

    # Список для накопления результатов
    results = []

    # Обрабатываем каждый год отдельно
    for year, year_data in bears_df.groupby('year'):
        coords = year_data[['latitude', 'longitude']].values

        # Если точек меньше, чем кластеров, пропускаем год
        if len(coords) < n_clusters:
            print(f"Предупреждение: в {year} году только {len(coords)} точек — кластеризация невозможна.")
            continue

        # Применяем K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(coords)
        centers = kmeans.cluster_centers_

        # Добавляем центры в результаты
        for cluster_id in range(n_clusters):
            results.append({
                'year': year,
                'cluster_id': cluster_id,
                'latitude': centers[cluster_id, 0],
                'longitude': centers[cluster_id, 1]
            })

    # Создаем DataFrame из результатов
    result_df = pd.DataFrame(results)

    # Сортируем по году и cluster_id для удобства
    result_df.sort_values(['year', 'cluster_id'], inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    return result_df

def process_cluster_centers(cluster_df):
    # 1. Удаляем колонку cluster_id
    df = cluster_df.drop(columns=['cluster_id'])

    # 2. Переименовываем координаты
    df = df.rename(columns={
        'latitude': 'latitude_bear',
        'longitude': 'longitude_bear'
    })

    # 3. Повторяем каждую строку 1000 раз
    df = df.loc[df.index.repeat(1000)].reset_index(drop=True)

    # 4. Генерируем колонку datetime
    datetimes = []
    for year in df['year'].unique():
        # Выбираем индексы строк с этим годом
        indices = df[df['year'] == year].index
        start_time = datetime(year, 1, 1, 0, 0, 0)
        for i, idx in enumerate(indices):
            dt = start_time + timedelta(hours=6 * i)
            datetimes.append(dt)

    # 5. Добавляем колонку datetime
    df['datetime'] = datetimes

    return df

def process_for_predict (ft_animal, ft_telemetry):
    merged = pd.merge(ft_animal, ft_telemetry, on='telemetry_id')
    result = merged[['timestamp', 'latitude', 'longitude']]
    res = get_bear_cluster_centers(result)
    result_df = process_cluster_centers(res)
    result_df['datetime'] = result_df['year'].apply(
        lambda year: datetime(year, 1, 1)  # Начинаем с 01-01-year
    ) + pd.to_timedelta(result_df.groupby('year').cumcount() * 2, unit='h')
    # Извлекаем временные признаки
    result_df['days_since_start'] = (result_df['datetime'] - result_df['datetime'].min()).dt.days
    result_df['month'] = result_df['datetime'].dt.month
    result_df['day'] = result_df['datetime'].dt.day
    result_df['hour'] = result_df['datetime'].dt.hour
    result_df['day_of_week'] = result_df['datetime'].dt.dayofweek
    return result_df
