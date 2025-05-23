import os
import pandas as pd
import geopandas as gpd
import random
from shapely.ops import linemerge, unary_union
import numpy as np
from shapely.geometry import MultiLineString, LineString
from scipy.ndimage import gaussian_filter1d


# Пути к файлам
ANIMALS_PATH = "C:/diplom/datasets/animals"
SHORELINES_PATH = "C:/diplom/datasets/shorelines"
ANIMALS_OUTPUT = "C:/diplom/datasets/animals/processed_animals.pkl"
SHORELINES_OUTPUT = "C:/diplom/datasets/shorelines/processed_shorelines.pkl"

# Функция обработки CSV-файлов с животными
def process_animal_datasets(path):
    all_data = []

    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            filepath = os.path.join(path, filename)
            df = pd.read_csv(filepath)

            # Приводим названия колонок к нижнему регистру
            df.columns = df.columns.str.lower()

            # Переименовываем datetime → date
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)

            # Приводим к типу date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

            # Оставляем только нужные колонки
            df = df[['date', 'latitude', 'longitude']].dropna()
            all_data.append(df)
            print(f"Process file: {filename}")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df =  final_df.drop_duplicates()
    print(f"Total rows: {len(final_df)}")
    
    return final_df

def animal_insert_data_clustered(animals_df, lat_min=69, lat_max=71, lon_min=-161, lon_max=-142, cluster_std=0.2):
    animals_df['date'] = pd.to_datetime(animals_df['date'])
    animals_df['year'] = animals_df['date'].dt.year

    min_year = animals_df['year'].min()
    max_year = animals_df['year'].max()
    all_years = set(range(min_year, max_year + 1))
    existing_years = set(animals_df['year'].unique())
    missing_years = sorted(all_years - existing_years)

    new_rows = []
    for year in missing_years:
        last_available_year = max(y for y in existing_years if y < year)
        mask = animals_df['year'] == last_available_year
        copied_data = animals_df[mask].copy()
        copied_data['date'] = copied_data['date'].apply(lambda x: x.replace(year=year))
        copied_data['year'] = year

        # Генерируем 2–3 случайных кластера для года
        num_clusters = random.choice([2, 3])
        cluster_centers = [
            (
                random.uniform(lat_min, lat_max),
                random.uniform(lon_min, lon_max)
            )
            for _ in range(num_clusters)
        ]

        # Назначаем каждой строке один из кластеров
        cluster_ids = np.random.choice(len(cluster_centers), size=len(copied_data))
        latitudes = []
        longitudes = []

        for cid in cluster_ids:
            lat_c, lon_c = cluster_centers[cid]
            # Добавляем гауссов шум (можно подрегулировать std)
            lat = np.clip(np.random.normal(lat_c, cluster_std), lat_min, lat_max)
            lon = np.clip(np.random.normal(lon_c, cluster_std), lon_min, lon_max)
            latitudes.append(lat)
            longitudes.append(lon)

        copied_data['latitude'] = latitudes
        copied_data['longitude'] = longitudes

        new_rows.append(copied_data)

    if new_rows:
        animals_df = pd.concat([animals_df] + new_rows, ignore_index=True)

    animals_df = animals_df.sort_values('date').reset_index(drop=True)
    animals_df = animals_df.drop(columns=['year'])

    return animals_df

def clusterize_bear_positions(df, lat_range=(69.5, 71), lon_range=(-159, -142), 
                               min_clusters=2, max_clusters=4, spread=0.4):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    result = []

    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year].copy()
        num_points = len(year_data)

        # Случайное число кластеров (2–4)
        num_clusters = random.randint(min_clusters, max_clusters)

        # Генерация центров кластеров
        cluster_centers = [
            (
                random.uniform(lat_range[0], lat_range[1]),
                random.uniform(lon_range[0], lon_range[1])
            )
            for _ in range(num_clusters)
        ]

        # Делим точки примерно поровну между кластерами
        cluster_sizes = np.random.multinomial(num_points, [1/num_clusters]*num_clusters)

        new_rows = []
        idx = 0
        for size, (center_lat, center_lon) in zip(cluster_sizes, cluster_centers):
            latitudes = np.random.normal(loc=center_lat, scale=spread, size=size)
            longitudes = np.random.normal(loc=center_lon, scale=spread, size=size)

            for lat, lon in zip(latitudes, longitudes):
                lat = np.clip(lat, lat_range[0], lat_range[1])
                lon = np.clip(lon, lon_range[0], lon_range[1])
                new_rows.append({
                    'date': year_data.iloc[idx]['date'],  # Сохраняем дату
                    'latitude': lat,
                    'longitude': lon
                })
                idx += 1

        result.extend(new_rows)

    # Собираем в итоговый DataFrame
    clustered_df = pd.DataFrame(result)
    return clustered_df

def adjust_coordinates(row, lat_min=69.5, lat_max=71, lon_min=-159, lon_max=-142):
    # Если широта выходит за пределы, заменяем на случайную точку в допустимом диапазоне
    if row['latitude'] < lat_min or row['latitude'] > lat_max or row['longitude'] < lon_min or row['longitude'] > lon_max:
        row['latitude'] = random.uniform(lat_min, lat_max)
        row['longitude'] = random.uniform(lon_min, lon_max)
    
    return row

def limit_bear_points_per_year(df, max_points=400):
    """
    Оставляет не более max_points записей на каждый год,
    случайным образом выбирая их из общего числа.
    """
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year

    limited_dfs = []
    for year, group in df.groupby('year'):
        if len(group) > max_points:
            group = group.sample(n=max_points, random_state=42)  # фиксируем seed для повторяемости
        limited_dfs.append(group)

    df_limited = pd.concat(limited_dfs, ignore_index=True)
    df_limited = df_limited.drop(columns='year')
    return df_limited

def load_and_combine_shapefiles(folder_path):
    # Находим все .shp файлы в указанной папке
    shapefiles = [f for f in os.listdir(folder_path) if f.endswith('.shp')]
    gdf_list = []

    for shapefile in shapefiles:
        shapefile_path = os.path.join(folder_path, shapefile)
        print(f"Loading {shapefile_path}...")
        
        # Читаем файл
        gdf = gpd.read_file(shapefile_path)
        
        # Оставляем только нужные колонки и переименовываем дату
        gdf = gdf[['DATE_', 'geometry']].rename(columns={'DATE_': 'date'})
        
        # Приводим дату к формату YYYY-MM-DD
        gdf['date'] = pd.to_datetime(gdf['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        gdf_list.append(gdf)
    
    # Объединяем все GeoDataFrame в один
    if gdf_list:
        gdf_combined = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
        gdf_combined['date'] = pd.to_datetime(gdf_combined['date'])
        print(f"Total files combined: {len(gdf_list)}")
        print(f"Total geometries in result: {len(gdf_combined)}")
        return gdf_combined
    else:
        print("No shapefiles found in the folder")
        return None

def second_processing (df1):
    df1['date'] = pd.to_datetime(df1['date'])
    df1['year'] = df1['date'].dt.year
    df1 = df1.drop(columns='date')
    # Группируем по году и объединяем геометрию
    grouped = df1.groupby('year')['geometry'].apply(lambda x: linemerge(unary_union(x)))

    # Превращаем обратно в GeoDataFrame
    df_yearly = gpd.GeoDataFrame(grouped, geometry='geometry')
    df_yearly = df_yearly.reset_index()
    # Заменим год на дату в формате datetime
    df_yearly['date'] = pd.to_datetime(df_yearly['year'].astype(str) + '-01-01')

    # Переупорядочим столбцы: сначала дата, потом геометрия
    df_yearly = df_yearly[['date', 'geometry']]
    return df_yearly

def simplify_multilinestring_to_linestring(gdf, tolerance=0.01):
    simplified_geoms = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        coords = []

        # Собираем все координаты из MultiLineString или LineString
        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                coords.extend(list(line.coords))
        elif isinstance(geom, LineString):
            coords.extend(list(geom.coords))
        else:
            continue  # Пропускаем, если геометрия другого типа

        # Убираем дубликаты и сортируем по долготе или широте (чтобы было похоже на контур)
        coords = list(dict.fromkeys(coords))  # Удаляем дубликаты, сохраняя порядок
        coords = sorted(coords, key=lambda x: (x[0], x[1]))  # Можно менять стратегию сортировки

        # Создаём линию и упрощаем её
        merged_line = LineString(coords).simplify(tolerance, preserve_topology=False)
        simplified_geoms.append({'date': row.date, 'geometry': merged_line})

    return gpd.GeoDataFrame(simplified_geoms, geometry='geometry', crs=gdf.crs)

def smooth_line(line, window_size=5):
    """
    Сглаживание линии с использованием скользящего среднего
    для уменьшения шумов в координатах.
    """
    coords = np.array(line.coords)

    # Если линия слишком короткая, не нужно сглаживать
    if len(coords) < window_size:
        return line

    # Применяем сглаживание с использованием скользящего среднего
    smoothed_coords = []

    for i in range(len(coords)):
        start_idx = max(i - window_size // 2, 0)
        end_idx = min(i + window_size // 2 + 1, len(coords))
        smoothed_coords.append(np.mean(coords[start_idx:end_idx], axis=0))

    smoothed_line = LineString(smoothed_coords)
    return smoothed_line

def smooth_lines_in_df(gdf, window_size=5):
    """
    Сглаживает все линии в DataFrame с помощью скользящего среднего
    """
    smoothed_geoms = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            smoothed_geom = smooth_line(geom, window_size)
        else:
            smoothed_geom = geom  # Если не линия, оставляем как есть

        smoothed_geoms.append({'date': row.date, 'geometry': smoothed_geom})

    return gpd.GeoDataFrame(smoothed_geoms, geometry='geometry', crs=gdf.crs)

def add_longitude_variation(geom, lon_min, lon_max,shift_range=(-0.5, 0.5)):
    """
    Слегка сдвигает долготные координаты точки в пределах указанного диапазона.
    """
    new_coords = []
    for x, y in geom.coords:
        shift = np.random.uniform(*shift_range)
        new_x = np.clip(x + shift, lon_min, lon_max)  # чтобы не выйти за пределы
        new_coords.append((new_x, y))
    return LineString(new_coords)

def resample_linestring(line: LineString, num_points: int) -> LineString:
    # Проверка на минимальное количество точек
    if len(line.coords) <= num_points:
        return line

    # Длина линии
    length = line.length

    # Расстояния, на которых нужно получить точки
    distances = np.linspace(0, length, num_points)

    # Получение точек вдоль линии
    new_coords = [line.interpolate(distance).coords[0] for distance in distances]

    return LineString(new_coords)

def shift_earliest_year_north(df, shift_value=1.5):
    df = df.copy()
    df['year'] = df['date'].dt.year
    min_year = df['year'].min()

    def shift_geometry_north(geom):
        coords = [(x, y + shift_value) for x, y in geom.coords]
        return LineString(coords)

    df.loc[df['year'] == min_year, 'geometry'] = df[df['year'] == min_year]['geometry'].apply(shift_geometry_north)
    df = df.drop(columns='year')
    return df

def generate_data(df, target_count=1000, start_year=1947, end_year=2018):
  df_res=df[(df['date'].dt.year == 1947) | (df['date'].dt.year == 1979)]
  # Уменьшение количества точек в геометрии
  df_res['geometry'] = df_res['geometry'].apply(lambda geom: resample_linestring(geom, target_count))
  df = df_res.sort_values('date').reset_index(drop=True)
  # Получаем первую геометрию
  base_line = df.loc[df['date'] == pd.Timestamp(f'{start_year}-01-01'), 'geometry'].values[0]

  # Генерация границ на каждый год
  generated_rows = []
  prev_line = base_line
  existing_years = set(df['date'].dt.year)
  for year in range(start_year + 1, end_year + 1):
    if year in existing_years:
        # Пропускаем год, если он уже есть в исходном DataFrame
        prev_line = df.loc[df['date'].dt.year == year, 'geometry'].values[0]
        continue

    prev_coords = np.array(prev_line.coords)
    n_points = prev_coords.shape[0]

    # Шум по долготе — широкий диапазон
    longitudes = prev_coords[:, 0] + np.random.uniform(-1.0, 1.0, size=n_points)

    # Смещение по широте: 70% вниз, 30% слегка колеблются
    lat_shift = np.zeros(n_points)
    indices = np.arange(n_points)
    np.random.shuffle(indices)

    down_idx = indices[:int(0.7 * n_points)]
    neutral_idx = indices[int(0.7 * n_points):]

    lat_shift[down_idx] = np.random.uniform(-0.001, -0.0005, size=down_idx.shape[0])
    lat_shift[neutral_idx] = np.random.uniform(-0.005, 0.005, size=neutral_idx.shape[0])

    latitudes = prev_coords[:, 1] + lat_shift

    # Сглаживание
    longitudes = gaussian_filter1d(longitudes, sigma=5)
    latitudes = gaussian_filter1d(latitudes, sigma=5)

    new_coords = list(zip(longitudes, latitudes))
    new_line = LineString(new_coords)

    generated_rows.append({
        'date': pd.Timestamp(f'{year}-01-01'),
        'geometry': new_line
    })

    # Обновляем prev_line
    prev_line = new_line

  # Создаем GeoDataFrame из новых данных
  generated_gdf = gpd.GeoDataFrame(generated_rows, crs=df.crs)

  # Объединяем с исходными данными
  final_gdf = pd.concat([df, generated_gdf]).sort_values('date').reset_index(drop=True)

  return final_gdf


# Запускаем обработку
if __name__ == "__main__":
    print("Process animal data...")
    df = process_animal_datasets(ANIMALS_PATH)
    temp_animal = df.apply(adjust_coordinates, axis=1)
    temp_df = animal_insert_data_clustered(temp_animal)
    animals_df = clusterize_bear_positions(temp_df)
    animals_df = limit_bear_points_per_year(animals_df)
    animals_df.to_pickle(ANIMALS_OUTPUT)
    print(f"Saved {len(animals_df)} rows of animal data in {ANIMALS_OUTPUT}")

    print("Process shorelines data...")
    df1 = load_and_combine_shapefiles(SHORELINES_PATH)
    df_yearly = second_processing(df1)
    df_simplified = simplify_multilinestring_to_linestring(df_yearly,tolerance=0.01)
    df_res = smooth_lines_in_df(df_simplified)
    df_res = shift_earliest_year_north(df_res) 
    df_boundary = generate_data(df_res)
    df_boundary.to_pickle(SHORELINES_OUTPUT)
    print(f"Saved {len(df_boundary)} rows of shorelines data in {SHORELINES_OUTPUT}")
 
