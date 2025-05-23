from ultralytics import YOLO
import cv2
import random
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import sys

model = YOLO(r"C:\diplom\python_files\prediction_animal\best.pt")  # загружаешь модель
video_path = r"C:\diplom\datasets\videos\video_3.mp4"
# video_path = sys.argv[1]

# Открываем видео
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Не удалось открыть видеофайл")

# Получаем параметры видео
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Значение по умолчанию
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps  # Длительность видео в секундах

# Фиксируем временные метки
start_time = datetime.now()
flight_id = int(start_time.strftime("%y%m%d%H%M"))
end_time = start_time + timedelta(seconds=duration)

# Инициализация данных
metadata = {
    'telemetry_id': [],
    'flight_id': [],
    'start_time': [],
    'end_time': [],
    'timestamp': [],
    'video_timestamp': [],
    'class_id': [],
    'class_name': [],
    'latitude': [],
    'longitude': [],
    'filename': [],
    'duration': []
}

# Генерация начальных координат
class_coords = {}
for class_id, class_name in model.names.items():
    class_coords[class_id] = {
        'longitude': random.uniform(-160, -145),
        'latitude': random.uniform(68, 71),
        'last_update': None
    }

# Переменные для отслеживания
telemetry_id = int((start_time.strftime("%y%m%d%H%M%S%f")[:-3]))
# telemetry_id = 1
last_detection_time = 0
current_frame = 0
detected_any = False

print(f"Начало обработки: {start_time}")
print(f"Длительность видео: {duration:.2f} секунд")
print(f"Конец полета: {end_time}")
print(f"Flight ID: {flight_id}")
print(f"Classes in model: {model.names}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1
    current_video_time = current_frame / fps

    # Детекция объектов
    results = model.predict(source=frame, save=False, conf=0.1)

    # Сбор классов
    detected_classes = set()
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            detected_classes.add(cls)
            detected_any = True

    # Запись метаданных только для polar_bear
    if current_video_time - last_detection_time >= 1.0 and detected_classes:
        detection_time = datetime.now().isoformat()

        for class_id in detected_classes:
            class_name = model.names[class_id]
            if class_name != 'polar_bear':
                continue  # Пропускаем все классы, кроме polar_bear

            # Обновляем координаты
            if class_coords[class_id]['last_update'] is None:
                class_coords[class_id].update({
                    'longitude': random.uniform(-160, -145),
                    'latitude': random.uniform(68, 71),
                    'last_update': current_video_time
                })
            else:
                class_coords[class_id]['longitude'] += random.uniform(-0.05, 0.05)
                class_coords[class_id]['latitude'] += random.uniform(-0.05, 0.05)
                class_coords[class_id]['last_update'] = current_video_time

            # Добавляем запись
            metadata['telemetry_id'].append(telemetry_id)
            metadata['flight_id'].append(flight_id)
            metadata['start_time'].append(start_time.isoformat())
            metadata['end_time'].append(end_time.isoformat())
            metadata['timestamp'].append(detection_time)
            metadata['video_timestamp'].append(current_video_time)
            metadata['class_id'].append(class_id)
            metadata['class_name'].append(class_name)
            metadata['latitude'].append(class_coords[class_id]['latitude'])
            metadata['longitude'].append(class_coords[class_id]['longitude'])
            metadata['filename'].append(video_path)
            metadata['duration'].append(duration)

            telemetry_id += 1

        last_detection_time = current_video_time

cap.release()

# Создаем DataFrame
metadata_df = pd.DataFrame(metadata)
print(metadata_df.head())

# # 1. ft_flight с уникальными значениями flight_id, start_time, end_time
# ft_flight = metadata_df[['flight_id', 'start_time', 'end_time']].drop_duplicates().reset_index(drop=True)

# # 2. ft_telemetry с колонками: telemetry_id, flight_id, timestamp, latitude, longitude
# ft_telemetry = metadata_df[['telemetry_id', 'flight_id', 'timestamp', 'latitude', 'longitude']].copy()

# # 3. ft_video с колонками: flight_id, filename, duration
# ft_video = metadata_df.groupby('flight_id')[['filename', 'duration']].first().reset_index()

# # 4. ft_animal с колонками: species (бывший class_name) и telemetry_id (только для polar_bear)
# ft_animal = metadata_df[metadata_df['class_name'] == 'polar_bear'][['class_name', 'telemetry_id']].rename(columns={'class_name': 'species'}).copy()

# if not metadata_df.empty:
#     print("\nУспешно обработано видео!")
#     print(f"Обнаружено записей: {len(metadata_df)}")
#     print(f"Из них polar_bear: {len(ft_animal)}")
# else:
#     print("\nПредупреждение: Не обнаружено ни одного животного!")

# # Вставка в БД
# DB_PATH = "C:/Users/12967/diplom.db"
# def create_tables(conn):
#     """Создаёт таблицы в БД, если они не существуют."""
#     cursor = conn.cursor()

#     # Таблица для полетов
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS ft_flight (
#             flight_id INTEGER PRIMARY KEY,
#             start_time timestamp,
#             end_time timestamp
#         )
#     """)

#     # Таблица для видео
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS ft_video (
#             video_id INTEGER PRIMARY KEY,
#             flight_id integer,  
#             filename TEXT,
#             duration REAL
#         )
#     """)

#     # Таблица для телеметрии
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS ft_telemetry (
#             telemetry_id INTEGER PRIMARY KEY,
#             flight_id integer,  
#             timestamp timestamp,
#             latitude REAL,
#             longitude REAL
#         )
#     """)
#     # Таблица для животных
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS ft_animal (
#             detection_id INTEGER PRIMARY KEY,
#             species TEXT,  
#             telemetry_id INTEGER
#         )
#     """)

#     conn.commit()

# def load_to_db():
#     conn = sqlite3.connect(DB_PATH)
#     create_tables(conn)

#     # Загрузка данных о полетах
#     print("Load ft_flight")
#     ft_flight.to_sql("ft_flight", conn, if_exists="append", index=False)
#     print(f"Loaded {len(ft_flight)} records in ft_flight")

#     # Загрузка данных о телеметрии
#     print("Load ft_telemetry")
#     ft_telemetry.to_sql("ft_telemetry", conn, if_exists="append", index=False)
#     print(f"Loaded {len(ft_telemetry)} records in ft_telemetry")

#     # Загрузка данных о видео
#     print("Load ft_video")
#     ft_video.to_sql("ft_video", conn, if_exists="append", index=False)
#     print(f"Loaded {len(ft_video)} records in ft_video")

#     # Загрузка данных о животных
#     print("Load ft_animal")
#     ft_animal.to_sql("ft_animal", conn, if_exists="append", index=False)
#     print(f"Loaded {len(ft_animal)} records in ft_animal")

#     conn.commit()
#     conn.close()
#     print("Success")

# load_to_db()

