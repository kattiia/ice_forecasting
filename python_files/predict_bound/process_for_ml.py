import pandas as pd
from shapely.geometry import LineString
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from shapely import wkt

def explode_geometry_to_points(df, step_hours=6):
    df['geometry'] = df['geometry'].apply(wkt.loads)
    exploded_rows = []

    for _, row in df.iterrows():
        base_date = pd.to_datetime(row['date'])
        line: LineString = row['geometry']
        coords = list(line.coords)
        num_points = len(coords)

        for i, (lon, lat) in enumerate(coords):
            timestamp = base_date + timedelta(hours=i * step_hours)
            exploded_rows.append({
                'datetime': timestamp,
                'latitude': lat,
                'longitude': lon
            })

    exploded_df = pd.DataFrame(exploded_rows)
    return exploded_df

def get_bear_cluster_centers(bears_df, n_clusters=3):
    # Извлекаем год из даты
    bears_df['date'] = pd.to_datetime(bears_df['date'])
    bears_df['year'] = bears_df['date'].dt.year

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

# Только для обучения
def process_for_ml(ice_df, bears_df):
  ice_points_df = explode_geometry_to_points(ice_df, step_hours=6)
  ice_points_df['year'] = ice_points_df['datetime'].dt.year
  clustered_df = get_bear_cluster_centers(bears_df)
  clustered_df['year'] = clustered_df['year']+1
  clustered_df = clustered_df[['year', 'latitude', 'longitude']]
  # Переименовываем колонки перед объединением
  ice_renamed = ice_points_df.rename(columns={
      'latitude': 'latitude_ice',
      'longitude': 'longitude_ice'
  })
  ice_renamed = ice_renamed[['year', 'latitude_ice', 'longitude_ice']]

  clustered_renamed = clustered_df.rename(columns={
      'latitude': 'latitude_bear',
      'longitude': 'longitude_bear'
  })
  # Объединяем по year (каждой точке льда сопоставляем ВСЕ кластеры медведей за тот же год)
  merged_df = pd.merge(
      ice_renamed,
      clustered_renamed,
      on='year',
      how='outer'  # Оставляем только общие года
  )
  # Сортируем по году и исходным индексам (для наглядности)
  merged_df = merged_df.sort_values(['year']).reset_index(drop=True)   
  merged_df['datetime'] = merged_df['year'].apply(
      lambda year: datetime(year, 1, 1)  # Начинаем с 01-01-year
  ) + pd.to_timedelta(merged_df.groupby('year').cumcount() * 2, unit='h')
  # Извлекаем временные признаки
  merged_df['days_since_start'] = (merged_df['datetime'] - merged_df['datetime'].min()).dt.days
  merged_df['month'] = merged_df['datetime'].dt.month
  merged_df['day'] = merged_df['datetime'].dt.day
  merged_df['hour'] = merged_df['datetime'].dt.hour
  merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek
  train_df = merged_df.dropna(subset=['latitude_ice', 'longitude_ice', 'latitude_bear', 'longitude_bear'])
  return train_df