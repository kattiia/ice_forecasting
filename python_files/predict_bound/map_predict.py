import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys


db_path = "C:/Users/12967/diplom.db"
conn = sqlite3.connect(db_path)
input_year = int(sys.argv[1])
# input_year = 2030
query_ice = f"""
SELECT * 
FROM ft_boundary_predict
WHERE year = '{input_year}'
"""
query_animal = f"""
SELECT t.latitude, t.longitude
FROM ft_animal a
JOIN ft_telemetry t ON a.telemetry_id = t.telemetry_id
WHERE strftime('%Y', t.timestamp) = '{input_year-5}' AND a.species = 'polar_bear'
union all
select latitude, longitude
from ft_history_animal
where strftime('%Y', date) = '{input_year-5}' 
"""

df_animal = pd.read_sql(query_animal, conn)
df_ice = pd.read_sql(query_ice, conn)

def plot_comparison(ice_df, title, animal_df, year):
    # Преобразуем геометрию льда из WKT в Shapely объекты
    ice_df['geometry'] = gpd.GeoSeries.from_wkt(ice_df['geometry'])
    
    # Создаем GeoDataFrame для льда
    gdf_ice = gpd.GeoDataFrame(ice_df, geometry='geometry', crs="EPSG:4326")
    
    # Создаем GeoDataFrame для животных
    geometry = [Point(xy) for xy in zip(animal_df['longitude'], animal_df['latitude'])]
    gdf_animals = gpd.GeoDataFrame(animal_df, geometry=geometry, crs="EPSG:4326")
    
    # Создаем карту
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Добавляем базовые элементы карты
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)
    
    # Отображаем границу льда
    for geom in gdf_ice['geometry']:
        if geom.geom_type == 'LineString':
            x, y = geom.xy
            ax.plot(x, y, color='purple', linewidth=2, 
                   transform=ccrs.PlateCarree(),
                   label=f'Граница льда {year+5} года')
        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                x, y = line.xy
                ax.plot(x, y, color='purple', linewidth=2, 
                       transform=ccrs.PlateCarree())
    
    # Отображаем животных
    if not gdf_animals.empty:
        ax.scatter(gdf_animals['longitude'], gdf_animals['latitude'],
                   color='red', s=10, transform=ccrs.PlateCarree(),
                   label=f'Полярные медведи ({year} год)')
    
    # Настройки графика
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.title(title)
    
    # Управление легендой (убираем дубликаты)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.grid(True)
    
    # Сохраняем
    plt.savefig("C:/diplom/python_files/predict_bound/predicted_ice_coordinates.png", 
               dpi=300, bbox_inches='tight')
    plt.close()

plot_comparison(
    ice_df=df_ice,
    title=f'Прогноз границы морского льда на {input_year} год',
    animal_df=df_animal,
    year=input_year-5
)