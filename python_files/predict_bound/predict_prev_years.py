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
input_year = 2022

db_path = "C:/Users/12967/diplom.db"
conn = sqlite3.connect(db_path)

query = f"""
SELECT * 
FROM ft_history_animal
WHERE strftime('%Y', date) = '{input_year}'
"""


ft_animal = pd.read_sql(query, conn)
ft_animal = ft_animal.rename(columns={
        'latitude': 'latitude_bear',
        'longitude': 'longitude_bear',
        'date':'datetime'
    })
ft_animal=ft_animal[['datetime', 'latitude_bear', 'longitude_bear']]
ft_animal['datetime'] = pd.to_datetime(ft_animal['datetime'])
ft_animal['year'] = ft_animal['datetime'].dt.year
ft_animal['days_since_start'] = (ft_animal['datetime'] - ft_animal['datetime'].min()).dt.days
ft_animal['month'] = ft_animal['datetime'].dt.month
ft_animal['day'] = ft_animal['datetime'].dt.day
ft_animal['hour'] = ft_animal['datetime'].dt.hour
ft_animal['day_of_week'] = ft_animal['datetime'].dt.dayofweek


def prepare_features(df, include_latice=False):
    features = ['latitude_bear', 'longitude_bear', 'days_since_start', 'month', 'day', 'hour']
    if include_latice:
        features.insert(2, 'latitude_ice')
    return df[features]

X_test_lat = prepare_features(ft_animal)
ft_animal['latitude_ice'] = model_lat.predict(X_test_lat)
X_test_lon = prepare_features(ft_animal, include_latice=True)
ft_animal['longitude_ice'] = model_lon.predict(X_test_lon)

# Создание таблицы, если она не существует
conn.execute("""
CREATE TABLE IF NOT EXISTS ft_boundary_predict (
    boundary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    geometry TEXT,
    year INTEGER
)
""")

# Преобразование координат в LineString и сохранение в БД
grouped = ft_animal.groupby('year')
insert_rows = []
title = input_year+5

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