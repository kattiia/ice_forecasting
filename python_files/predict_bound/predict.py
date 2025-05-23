import sqlite3
import pandas as pd
import joblib
from process_for_predict import process_for_predict
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.wkt import dumps as wkt_dumps

model_lat = joblib.load(r"C:\diplom\python_files\predict_bound\model_lat.pkl")
model_lon = joblib.load(r"C:\diplom\python_files\predict_bound\model_lon.pkl")
input_year = 2025

db_path = "C:/Users/12967/diplom.db"
conn = sqlite3.connect(db_path)

query_telemetry = f"""
SELECT * 
FROM ft_telemetry
WHERE strftime('%Y', timestamp) = '{input_year}'
"""

query_animal = f"""
SELECT a.*
FROM ft_animal a
JOIN ft_telemetry t ON a.telemetry_id = t.telemetry_id
WHERE strftime('%Y', t.timestamp) = '{input_year}'
"""

query = f"""
SELECT t.latitude, t.longitude
FROM ft_animal a
JOIN ft_telemetry t ON a.telemetry_id = t.telemetry_id
WHERE strftime('%Y', t.timestamp) = '{input_year}' AND a.species = 'polar_bear'
"""

ft_animal = pd.read_sql(query_animal, conn)
ft_telemetry = pd.read_sql(query_telemetry, conn)
df = pd.read_sql(query, conn)


def plot_comparison(test_df, title, df_animals, year):
    # Группируем по году и создаем линии
    grouped = test_df.groupby('year')
    result = []
    
    for year, group in grouped:
        end_points = [Point(lon, lat) for lon, lat in zip(group['longitude_ice'], group['latitude_ice'])]
        end_line = LineString(end_points)
        result.append({'year': year, 'end_line': end_line})
    
    # Создаем GeoDataFrame (хотя в данном случае можно обойтись и без него)
    gdf = gpd.GeoDataFrame(result, geometry='end_line')
    
    # Создаем карту
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Добавляем границы стран, океаны и другие элементы
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)

    # Отображаем животных
    if not df_animals.empty:
        ax.scatter(df_animals['longitude'], df_animals['latitude'],
                   color='red', s=10, transform=ccrs.PlateCarree(),
                   label=f'Животные в {year} году')
    
    # Рисуем линии (фиолетовые)
    for year, line in zip(gdf['year'], gdf['end_line']):
        x, y = line.xy
        ax.plot(x, y, 
                transform=ccrs.PlateCarree(), 
                color='purple',  # Фиолетовый цвет
                linewidth=2, 
                label=str(year+5))
    
    # Настройки графика
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Сохраняем
    plt.savefig("C:/diplom/python_files/predict_bound/predicted_ice_coordinates.png", 
                dpi=300, bbox_inches='tight')


def prepare_features(df, include_latice=False):
    features = ['latitude_bear', 'longitude_bear', 'days_since_start', 'month', 'day', 'hour']
    if include_latice:
        features.insert(2, 'latitude_ice')
    return df[features]

test_df = process_for_predict(ft_animal, ft_telemetry)
plot_df_bear = df[['latitude', 'longitude']]
now_year = int(input_year)
title = now_year+5
X_test_lat = prepare_features(test_df)
test_df['latitude_ice'] = model_lat.predict(X_test_lat)
X_test_lon = prepare_features(test_df, include_latice=True)
test_df['longitude_ice'] = model_lon.predict(X_test_lon)


# plot_comparison(
#     test_df=test_df,
#     title=f'Прогноз границы морского льда на {title} год',
#     df_animals=plot_df_bear,
#     year=now_year
# )

# Создание таблицы, если она не существует
conn.execute("""
CREATE TABLE IF NOT EXISTS ft_boundary_predict (
    boundary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    geometry TEXT,
    year INTEGER
)
""")

# Преобразование координат в LineString и сохранение в БД
grouped = test_df.groupby('year')
insert_rows = []

for year, group in grouped:
    line = LineString(zip(group['longitude_ice'], group['latitude_ice']))
    wkt_line = wkt_dumps(line)
    insert_rows.append((wkt_line, year + 5))  # Прогноз делается на 5 лет вперёд

# Удаляем старые записи на тот же год (чтобы не дублировать)
conn.execute("DELETE FROM ft_boundary_predict WHERE year = ?", (title,))
print(f"🗑 Удалены предыдущие записи для года {title} из ft_boundary_predict.")

# Сохраняем в таблицу
conn.executemany("INSERT INTO ft_boundary_predict (geometry, year) VALUES (?, ?)", insert_rows)
conn.commit()
print(f"✅ Сохранено {len(insert_rows)} записей в таблицу ft_boundary_predict.")