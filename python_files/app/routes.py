from flask import Flask, render_template, request, redirect, url_for, send_file
from app.animal_map import plot_animal_and_ice_boundary
import os
from datetime import datetime
import subprocess
import gdown
import tempfile
import re

app = Flask(__name__, template_folder='../templates', static_folder='../static')
db_path = r"C:/Users/12967/diplom.db"

current_year = 2019  # значение по умолчанию

def download_from_gdrive(gdrive_url):
    file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', gdrive_url)
    if not file_id_match:
        return None
    file_id = file_id_match.group(1)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", temp_file.name, quiet=False)
        return temp_file.name
    except Exception as e:
        print("Ошибка при загрузке файла:", e)
        return None
    

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_year
    if request.method == 'POST':
        if 'year' in request.form:
            year = int(request.form['year'])
            current_year = year
            plot_animal_and_ice_boundary(year, db_path)
        elif 'refresh' in request.form:
            pass  # просто обновим картинку
        return redirect(url_for('index'))
    return render_template('index.html', year=current_year)

@app.route('/image')
def image():
    image_path = r"C:\diplom\python_files\app\animal_and_ice.png"
    if not os.path.exists(image_path):
        return "Изображение не найдено", 404
    return send_file(image_path, mimetype='image/png', last_modified=datetime.utcnow())

# Страница прогноза
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'run_prediction':
            script_path = r"C:\diplom\python_files\predict_bound\predict.py"
            subprocess.run(['python', script_path], check=True)

        elif action == 'upload_uav':
            video_url = request.form.get('video_url')
            local_path = download_from_gdrive(video_url)
            if local_path:
                script_path = r"C:\diplom\python_files\prediction_animal\animal_from_video.py"
                subprocess.run(['python', script_path, local_path], check=True)
            else:
                print("Не удалось загрузить видео.")

        elif action == 'show_forecast':
            year = int(request.form.get('forecast_year'))
            script_path = r"C:\diplom\python_files\predict_bound\map_predict.py"
            subprocess.run(['python', script_path, str(year)], check=True)

    return render_template('forecast.html', time_now=datetime.now().timestamp())

# Отображение картинки прогноза
@app.route('/predicted_image')
def predicted_image():
    image_path = r"C:\diplom\python_files\predict_bound\predicted_ice_coordinates.png"
    return send_file(image_path, mimetype='image/png')