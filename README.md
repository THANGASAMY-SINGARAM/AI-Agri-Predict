# AgriPredict

AgriPredict is a full-stack AI agriculture assistant built with Streamlit, MongoDB, and machine learning models. It brings multiple student-level AI components into one working application: crop yield prediction, soil and crop image classification, fertilizer estimation, multilingual support, voice assistance, authentication, chatbot guidance, and prediction history.

This project is designed as a practical AIML learning project, not as a production agricultural platform. It demonstrates how machine learning, database-backed user workflows, and an interactive web interface can work together in a single applied AI system.

## Project Highlights

- End-to-end Streamlit application with login and dashboard flow
- MongoDB integration for users and crop yield records
- Crop yield prediction using trained machine learning models
- Soil and crop image classification using TensorFlow/Keras models
- Fertilizer recommendation with NPK quantity and cost estimation
- Multilingual UI labels for better accessibility
- Voice input and voice output support
- Rule-based agriculture chatbot for common farming guidance
- Database tab for importing and reviewing yield records

## Why This Project Matters

Most beginner AIML projects stop after training a model and displaying accuracy. AgriPredict goes further by connecting model predictions with user authentication, database storage, multiple workflows, and a usable web interface.

For interviews and placement discussions, this project shows:

- Ability to build a complete AI-powered application
- Understanding of model loading and prediction pipelines
- Database usage beyond static CSV files
- Full-stack thinking with UI, backend logic, storage, and ML integration
- Practical debugging and extension opportunities

## Tech Stack

| Area | Technologies |
| --- | --- |
| Frontend and UI | Streamlit |
| Programming | Python |
| Machine Learning | scikit-learn, TensorFlow/Keras |
| Data Processing | pandas, NumPy |
| Database | MongoDB, PyMongo |
| Voice Features | SpeechRecognition, gTTS |
| Model Storage | Pickle, H5 |

## System Architecture

```text
User
  |
  v
Streamlit UI
  |
  +-- Authentication flow
  |     |
  |     v
  |   MongoDB users collection
  |
  +-- Yield prediction form
  |     |
  |     v
  |   crop_yield_model.pkl
  |
  +-- Soil/crop image upload
  |     |
  |     v
  |   TensorFlow image classifier models
  |
  +-- Fertilizer planner
  |     |
  |     v
  |   NPK and cost calculation logic
  |
  +-- Chatbot and voice support
  |
  v
MongoDB yield records collection
```

## Project Structure

```text
.
|-- app.py                 Main Streamlit application
|-- database.py            MongoDB connection, users, and yield records
|-- i18n.py                Language labels and translation helpers
|-- init_database.py       Database initialization helper
|-- train_yield_model.py   Yield model training script
|-- requirements.txt       Python dependencies
|-- sample_data.csv        Sample crop yield dataset
|-- dataset/               Local image dataset folders
|-- crop_yield_model.pkl   Trained yield prediction model
|-- soil_model.h5          Trained soil classifier model
`-- crop_model.h5          Trained crop classifier model
```

Large model and dataset files are intentionally kept out of GitHub. Place them in the project root with the names shown above before running the full application.

## Core Modules

### 1. Authentication

The app includes user registration and login. User credentials are stored in MongoDB using salted password hashing instead of plain text passwords.

### 2. Yield Prediction

The yield prediction workflow accepts agricultural inputs such as temperature, rainfall, irrigation, fertilizer quantity, and soil nutrients. These values are converted into a model input row and passed to the trained yield model.

### 3. Image Classification

The image classifier accepts soil or crop images, preprocesses them, and predicts the class using TensorFlow/Keras models.

### 4. Fertilizer Recommendation

The fertilizer module estimates NPK requirements, fertilizer bag counts, and approximate cost based on crop, soil, and land area.

### 5. Multilingual and Voice Support

The UI uses a translation helper for labels and includes optional speech input and text-to-speech output.

### 6. Chatbot

The chatbot is rule-based and provides practical guidance on crop selection, fertilizer, irrigation, soil health, pests, and yield improvement.

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

## MongoDB Configuration

By default, the app connects to:

```text
mongodb://localhost:27017
```

You can override the MongoDB settings:

```powershell
$env:AGRIPREDICT_MONGO_URI="mongodb://localhost:27017"
$env:AGRIPREDICT_DB_NAME="agripredict"
```

## Database Setup

Start MongoDB locally, then initialize the database:

```powershell
python init_database.py
```

This prepares indexes and imports sample records when `sample_data.csv` is available.

## Train The Yield Model

```powershell
python train_yield_model.py
```

The script reads yield records from MongoDB and saves the trained model as `crop_yield_model.pkl`.

## Run The Application

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, create an account, and sign in to use the dashboard.

## Required Model Files

The complete app expects these files in the project root:

- `crop_yield_model.pkl`
- `soil_model.h5`
- `crop_model.h5`

If TensorFlow or the `.h5` files are missing, image classification will be disabled, but other modules can still run.

## Interview Talking Points

If asked to explain this project, focus on:

- Why MongoDB was used for flexible user and yield record storage
- How `app.py` controls the Streamlit UI and user workflows
- How `database.py` separates database operations from UI logic
- How model files are loaded once and reused for predictions
- How image preprocessing is done before classification
- Why multilingual fallback handling prevents UI crashes
- What limitations exist compared to production agriculture platforms

## Current Limitations

- Uses local/static datasets instead of live agricultural data feeds
- Does not yet include live weather APIs or satellite imagery
- Image model quality depends on the available training dataset
- Chatbot is rule-based, not an LLM-powered assistant
- No mobile app version yet
- Not designed for production-scale deployment

## Future Improvements

- Add live weather API integration
- Add satellite or remote sensing data support
- Improve model evaluation reports and accuracy tracking
- Add farmer-specific prediction history dashboards
- Deploy with cloud database and authentication services
- Add a mobile-first interface
- Integrate a retrieval-based or LLM-powered agriculture assistant

## Disclaimer

AgriPredict is built for learning, demonstration, and planning support. Fertilizer use, pesticide decisions, disease treatment, and crop planning should be confirmed with local agriculture officers, soil-test labs, or qualified agronomists before real field use.

## License

This project is licensed under the terms included in the `LICENSE` file.
