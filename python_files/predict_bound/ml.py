import pandas as pd
from shapely.geometry import LineString
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3
from process_for_ml import process_for_ml 

db_path = "C:/Users/12967/diplom.db"
conn = sqlite3.connect(db_path)

ice_df = pd.read_sql("SELECT * FROM ft_boundary", conn)
bear_df = pd.read_sql("SELECT * FROM ft_history_animal", conn)

train_df = process_for_ml(ice_df, bear_df)
def prepare_features(df, include_latice=False):
    features = ['latitude_bear', 'longitude_bear', 'days_since_start', 'month', 'day', 'hour']
    if include_latice:
        features.insert(2, 'latitude_ice')
    return df[features]


def evaluate_model(y_true, y_pred, name=""):
    print(f"\n📊 Оценка модели ({name}):")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.5f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.5f}")
    print(f"R²:   {r2_score(y_true, y_pred):.5f}")


def plot_comparison(lon_true, lat_true, lon_pred, lat_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(lon_true, lat_true, color='red', label='Фактические координаты льда', alpha=0.6)
    plt.scatter(lon_pred, lat_pred, color='blue', label='Предсказанные координаты льда', alpha=0.6)
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def tune_model(X, y, model_name=""):
    print(f"\n🔍 Поиск лучших параметров для модели ({model_name})...")
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7]
    }

    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    print("✅ Лучшие параметры:", grid_search.best_params_)

    # Кросс-валидация с лучшей моделью
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
    print(f"📈 Средний R² по кросс-валидации: {cv_scores.mean():.5f}")

    return best_model

# === Обучение модели широты ===
X_lat = prepare_features(train_df)
y_lat = train_df['latitude_ice']

model_lat = tune_model(X_lat, y_lat, model_name="Широта")
joblib.dump(model_lat, "model_lat.pkl")

# === Обучение модели долготы ===
X_lon = prepare_features(train_df, include_latice=True)
y_lon = train_df['longitude_ice']

model_lon = tune_model(X_lon, y_lon, model_name="Долгота")
joblib.dump(model_lon, "model_lon.pkl")