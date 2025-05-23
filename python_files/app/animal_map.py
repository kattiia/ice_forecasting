import sqlite3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely import wkt
from sys import argv
import pandas as pd


def plot_animal_and_ice_boundary(year, db_path):
    # Подключаемся к базе данных
    conn = sqlite3.connect(db_path)

    # Загружаем координаты животных
    query_animals = """
    SELECT latitude, longitude 
    FROM ft_history_animal 
    WHERE strftime('%Y', date) = ?
    """
    df_animals = pd.read_sql_query(query_animals, conn, params=(str(year),))

    # Загружаем геометрию ледяной границы
    query_boundary = """
    SELECT geometry 
    FROM ft_boundary 
    WHERE strftime('%Y', date) = ?
    """
    df_boundary = pd.read_sql_query(query_boundary, conn, params=(str(year),))

    # Закрываем соединение с БД
    conn.close()

    if df_animals.empty and df_boundary.empty:
        print(f"Нет данных о животных и границах за {year} год.")
        return

    # Преобразуем геометрию в GeoDataFrame
    if not df_boundary.empty:
        df_boundary['geometry'] = df_boundary['geometry'].apply(wkt.loads)
        gdf_boundary = gpd.GeoDataFrame(df_boundary, geometry='geometry', crs='EPSG:4326')
    else:
        gdf_boundary = None

    # Создаем карту
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Базовые элементы карты
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Отображаем животных
    if not df_animals.empty:
        ax.scatter(df_animals['longitude'], df_animals['latitude'],
                   color='red', s=10, transform=ccrs.PlateCarree(),
                   label=f'Животные в {year} году')

    # Отображаем границы льда
    if gdf_boundary is not None:
        gdf_boundary.plot(ax=ax, color='purple', linewidth=2, transform=ccrs.PlateCarree(), label='Ледяная граница')

    # Настройки
    ax.set_title(f'Местоположение животных и границы льда за {year} год')
    ax.legend()
    ax.gridlines()

    output_file = r"C:\diplom\python_files\app\animal_and_ice.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Карта сохранена как {output_file}")

if __name__ == "__main__":
    if len(argv) != 2:
        print("Использование: python script.py <год>")
    else:
        try:
            year = int(argv[1])
            db_path = r"C:/Users/12967/diplom.db"
            plot_animal_and_ice_boundary(year, db_path)
        except ValueError:
            print("Пожалуйста, введите корректный год (число)")
