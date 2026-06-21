# AgriPredict

AgriPredict is an AI-assisted agriculture decision support application built with Streamlit. It combines crop yield prediction, soil and crop image classification, fertilizer planning, multilingual UI labels, voice support, authentication, and a local agriculture chatbot in one dashboard.

## Features

- Crop yield prediction from weather, irrigation, fertilizer, and soil nutrient inputs
- Soil and crop image classification with TensorFlow/Keras models
- Fertilizer planning with NPK quantity, bag count, and estimated cost
- Rule-based agriculture chatbot for common farming questions
- User registration and login with salted password hashing
- MongoDB-backed storage for users and crop yield records
- Multilingual interface support for English and several Indian languages
- Optional speech input and voice output support

## Tech Stack

- Python
- Streamlit
- TensorFlow/Keras
- scikit-learn
- pandas and NumPy
- MongoDB with PyMongo
- SpeechRecognition
- gTTS

## Project Structure

```text
.
├── app.py                 # Main Streamlit application
├── database.py            # MongoDB connection, users, and yield records
├── i18n.py                # Language labels and translation helpers
├── init_database.py       # Database initialization and sample import helper
├── train_yield_model.py   # Yield model training script
├── requirements.txt       # Python dependencies
├── sample_data.csv        # Local sample yield dataset
├── dataset/               # Local image dataset folders
├── crop_yield_model.pkl   # Trained yield model artifact
├── soil_model.h5          # Trained soil classifier artifact
└── crop_model.h5          # Trained crop classifier artifact
```

Large model and dataset files are usually kept outside GitHub. Place them in the project root using the names shown above before running the full app.

## Prerequisites

- Python 3.10 or newer
- MongoDB running locally or a reachable MongoDB connection URI
- A microphone setup if speech input is required

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Configuration

By default, the app connects to MongoDB at:

```text
mongodb://localhost:27017
```

You can override the database configuration with environment variables:

```powershell
$env:AGRIPREDICT_MONGO_URI="mongodb://localhost:27017"
$env:AGRIPREDICT_DB_NAME="agripredict"
```

## Database Setup

Start MongoDB, then initialize the database:

```powershell
python init_database.py
```

This prepares indexes and imports records from `sample_data.csv` when available. The app also provides a Database tab for importing sample records.

## Training The Yield Model

Train or refresh the crop yield model with:

```powershell
python train_yield_model.py
```

The script reads records from MongoDB and saves the trained model as `crop_yield_model.pkl`.

## Running The App

Start the Streamlit application:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, create an account, and sign in to access the dashboard.

## Model Files

The app expects these model artifacts in the project root:

- `crop_yield_model.pkl`
- `soil_model.h5`
- `crop_model.h5`

If TensorFlow is not installed or the `.h5` files are missing, image classification will be unavailable, but the rest of the app can still run.

## Notes

AgriPredict is intended for educational and planning use. Fertilizer recommendations, pesticide use, disease treatment, and crop decisions should be verified with local agriculture officers, soil-test labs, or qualified agronomists before field application.

## License

This project is licensed under the terms included in the `LICENSE` file.
