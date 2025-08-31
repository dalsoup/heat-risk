# Heat Risk Mini

## 0) 설치
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy requests fastapi uvicorn xgboost scikit-learn joblib python-dotenv pyarrow certifi

## 1) .env / admin_dong.csv 준비
- .env: KMA_SERVICE_KEY=... (미인코딩 또는 1회 인코딩)
- 진단시만 KMA_USE_HTTP=true

## 2) 수집 → 피처 → 라벨 → 학습
python fetch_weather.py
python build_features.py
python prepare_labels.py
python train_hourly.py

## 3) 서빙
uvicorn serve_api:app --reload --port 8080
curl "http://127.0.0.1:8080/predict?hour=0&drink=2&shelter_min=45&indoor_min=0"
