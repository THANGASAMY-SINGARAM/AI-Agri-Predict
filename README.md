# AgriPredict

AgriPredict is a Streamlit app for crop yield prediction, soil/crop image
classification, fertilizer planning, and an agriculture chatbot for quick
farm guidance.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Microphone input needs a working local microphone setup. If `SpeechRecognition`
cannot access the microphone, install and configure PyAudio for your Python
version.

## Run The App

```powershell
streamlit run app.py
```

## MongoDB

AgriPredict stores login users and yield records in a local MongoDB database.
Start MongoDB locally before opening the app. By default the app connects to:

```text
mongodb://localhost:27017
```

You can override the connection with:

```powershell
$env:AGRIPREDICT_MONGO_URI="mongodb://localhost:27017"
$env:AGRIPREDICT_DB_NAME="agripredict"
```

## Initialize The Database

```powershell
python init_database.py
```

This imports rows from `sample_data.csv` into MongoDB. The app can also import
the CSV from the Database tab.

## Train The Yield Model

```powershell
python train_yield_model.py
```

The script reads training rows from MongoDB. If the database is empty,
it imports `sample_data.csv` first. The trained model is saved as
`crop_yield_model.pkl`.

## Image Models

The app loads:

- `soil_model.h5`
- `crop_model.h5`

Classifier labels are read from the folders inside `dataset/soil` and
`dataset/crop`, sorted alphabetically. Keep the same folder structure when
training and running the models.

## Notes

The agriculture chatbot is rule-based and runs locally. It can answer common
questions about crop selection, fertilizer, irrigation, soil health, pests, and
yield improvement, but local expert guidance should be used for chemical doses
and disease treatment.

The fertilizer planner gives planning estimates only. Final fertilizer rates
should be confirmed with local soil-test advice.
