# CodTech Python Internship — All 4 Tasks
**Intern:** Sudeep  
**Domain:** Python Programming  
**Organization:** CodTech IT Solutions

---

## 📁 Project Structure

```
codtech-python-internship/
├── task1_api_visualization.py     # Task 1: Weather API + Dashboard
├── task2_report_generation.py     # Task 2: Automated PDF Report
├── task3_chatbot.py               # Task 3: AI Chatbot with NLP
├── task4_ml_model.py              # Task 4: ML Spam Detection
├── requirements.txt               # All dependencies
└── README.md                      # This file
```

---

## ✅ Task 1 — API Integration & Data Visualization

**Description:** Fetches real-time weather data for 10 Indian cities using the OpenWeatherMap API and creates a 6-chart visualization dashboard.

**Features:**
- Live weather data via REST API (with fallback sample data)
- 6 charts: temperature bar, humidity pie, wind speed horizontal bar, weather heatmap, min-max range, pressure vs cloudiness scatter
- Dark-themed dashboard saved as PNG
- Data exported to CSV

**Run:**
```bash
python task1_api_visualization.py
```
> To use real data, replace `API_KEY` in the script with your key from [openweathermap.org](https://openweathermap.org/api)

**Output:** `task1_weather_dashboard.png`, `task1_weather_data.csv`

---

## ✅ Task 2 — Automated Report Generation

**Description:** Reads student performance data, performs statistical analysis, and generates a professional multi-page PDF report with charts and tables.

**Features:**
- Auto-generated realistic student dataset (60 students, 5 subjects)
- 4 embedded charts: department averages, grade pie, subject boxplots, attendance scatter
- 3-page PDF: cover + summary, visual analysis, top 10 students table
- Department-wise summary table

**Run:**
```bash
python task2_report_generation.py
```

**Output:** `task2_student_report.pdf`, `task2_student_data.csv`

---

## ✅ Task 3 — AI Chatbot with NLP

**Description:** A college information chatbot using NLTK for text preprocessing, TF-IDF for intent matching, and a Streamlit web UI with conversation memory.

**Features:**
- 13 intent categories (admission, fees, placement, hostel, etc.)
- NLP pipeline: tokenization → lemmatization → stopword removal → TF-IDF cosine similarity
- Conversation history / memory
- Quick-topic buttons in sidebar
- Streamlit UI + terminal fallback

**Run (Streamlit UI):**
```bash
streamlit run task3_chatbot.py
```

**Run (Terminal):**
```bash
python task3_chatbot.py
```

---

## ✅ Task 4 — Machine Learning Model Implementation

**Description:** Spam email detection using 5 ML models with full EDA, cross-validation, hyperparameter tuning, ROC curves, and confusion matrix.

**Features:**
- 5 models compared: Logistic Regression, Naive Bayes, Linear SVM, Random Forest, Gradient Boosting
- TF-IDF with bigrams (n-gram range 1–2)
- 5-fold cross-validation for all models
- GridSearchCV hyperparameter tuning
- EDA: class distribution, message length, word count density
- Evaluation: accuracy comparison, confusion matrix, ROC curves
- Live prediction demo on new emails
- Model saved with joblib

**Run:**
```bash
python task4_ml_model.py
```

**Output:** `task4_evaluation.png`, `task4_eda.png`, `task4_spam_model.pkl`, `task4_email_dataset.csv`

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🛠 Technologies Used

| Library | Purpose |
|---|---|
| `requests` | API calls |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualizations |
| `streamlit` | Web UI for chatbot |
| `nltk` / `scikit-learn` | NLP & ML |
| `reportlab` | PDF generation |
| `joblib` | Model serialization |

---

## 📜 License
This project was built as part of the CodTech Python Internship program.
