import sqlite3
import pandas as pd
import pickle
from shapely.wkt import dumps

# Пути к БД и файлам
DB_PATH = "C:/Users/12967/diplom.db"
ANIMALS_FILE = "C:/diplom/datasets/animals/processed_animals.pkl"
SHORELINES_FILE = "C:/diplom/datasets/shorelines/processed_shorelines.pkl"

def create_tables(conn):
    """Создаёт таблицы в БД, если они не существуют."""
    cursor = conn.cursor()

    # Таблица для животных
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ft_history_animal (
            animal_history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date date,
            latitude REAL,
            longitude REAL
        )
    """)

    # Таблица для береговой линии (с WKT)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ft_boundary (
            boundary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            geometry TEXT,  
            date date
        )
    """)

    conn.commit()

def load_to_db():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    # # Загрузка данных о животных (без изменений)
    print("Load animal data in ft_history_animal...")
    animals_df = pd.read_pickle(ANIMALS_FILE)
    animals_df.to_sql("ft_history_animal", conn, if_exists="append", index=False)
    print(f"Loaded {len(animals_df)} records in ft_history_animal")

    # Загрузка данных о границах с преобразованием LineString → WKT
    print("Load shorelines data in ft_boundary...")
    shorelines_df = pd.read_pickle(SHORELINES_FILE)
    
    # Преобразуем геометрию в WKT (если столбец называется 'geometry')
    if 'geometry' in shorelines_df.columns:
        shorelines_df['geometry'] = shorelines_df['geometry'].apply(dumps)  # Shapely → WKT
    
    # Вставляем в БД
    shorelines_df.to_sql("ft_boundary", conn, if_exists="append", index=False)
    print(f"Loaded {len(shorelines_df)} records in ft_boundary")
    
    conn.commit()
    conn.close()
    print("Success")

if __name__ == "__main__":
    load_to_db()
