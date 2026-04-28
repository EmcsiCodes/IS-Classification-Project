# IS Classification Project

Machine learning classification project based on synthetic student-AI interaction sessions.

## Dataset Source

This project uses the dataset from Kaggle:
https://www.kaggle.com/datasets/ayeshasal89/ai-assistant-usage-in-student-life-synthetic

## About Dataset

**AI Assistant Usage in Student Life**

Explore how students at different academic levels use AI tools like ChatGPT for tasks such as coding, writing, studying, and brainstorming. It is designed for learning, exploratory data analysis (EDA), and ML experimentation.

This dataset simulates **10,000 sessions** of students interacting with an AI assistant. Each row represents one session and includes student level, discipline, task type, session length, AI effectiveness, satisfaction rating, and whether the student reused the tool.

All data is **synthetically generated** using controlled distributions, real-world logic, and behavioral modeling to reflect realistic usage patterns.

## Why This Dataset Was Created

As AI tools become mainstream in education, there is a need to analyze and model student interaction behavior. Public datasets for this are limited, so this synthetic dataset supports:

- EDA and visualization practice
- Machine learning modeling
- Feature engineering workflows
- Educational data science exploration

It is suitable for students, data science learners, and researchers who want realistic use cases without privacy or copyright issues.

## Dataset Structure

Source file: `ai_assistant_usage_student_life.csv` (original raw CSV). For the project work we use a cleaned and feature-engineered copy saved at [data/processed.csv](data/processed.csv).

Key columns (summary):

- `SessionID`: unique session identifier
- `StudentLevel`: academic level (High School, Undergraduate, Graduate)
- `Discipline`: field of study (Computer Science, Psychology, ...)
- `SessionDate`, `SessionMonth`, `SessionDay`
- `SessionLengthMin`, `TotalPrompts`, `Promts_Per_Min`
- `TaskType`, `AI_AssistanceLevel`, `SatisfactionRating`
- `FinalOutcome`: multi-class session outcome (Assignment Completed, Idea Drafted, Confused, Gave Up)
- `UsedAgain`: binary target (whether the student reused the assistant)

## Notebooks & Artifacts

- EDA & preprocessing: [src/eda.ipynb](src/eda.ipynb) and [src/preprocessing.ipynb](src/preprocessing.ipynb) — exploratory analysis and deterministic feature engineering that produces `data/processed.csv`.
- Modeling (UsedAgain): [src/modelling_usedAgain.ipynb](src/modelling_usedAgain.ipynb) — readable pipeline that trains several classifiers, includes an optional training-cell to balance the `UsedAgain` training set (RandomOverSampler or sklearn fallback), and saves models to `models/usedAgain/`.
- Modeling (FinalOutcome): [src/modelling_finalOutcome.ipynb](src/modelling_finalOutcome.ipynb) — multi-class training workflow using the same engineered features; saves models to `models/finalOutcome/`.
- Models/artifacts: trained model joblib files are stored in `models/usedAgain/` and `models/finalOutcome/`. Consider persisting the fitted preprocessor as `models/preprocessor.joblib` for inference consistency.

## Quick Start

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Open the notebooks in `src/` and run cells in order. To inspect the processed dataset directly:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed.csv'); print(df.shape); print(df.columns.tolist()[:15])"
```

3. Run the modelling notebooks to reproduce training and save artifacts. Model files will be written to `models/usedAgain/` and `models/finalOutcome/`.

## What is implemented

- Deterministic preprocessing that produces `data/processed.csv` with engineered features (prompt density, duration bins, ordinal encodings, etc.).
- Modeling notebooks that train and compare multiple classifiers: Logistic Regression, Random Forest, KNN, SVM, GaussianNB, AdaBoost (and baselines). Training includes stratified splits and handling for class imbalance (`class_weight='balanced'` where supported, and oversampling via `imblearn.RandomOverSampler` or sklearn fallback).
- Evaluation tables showing accuracy plus macro-averaged Precision/Recall/F1 to account for class imbalance.
- Model artifact saving via `joblib.dump()` into the `models/` directory.
