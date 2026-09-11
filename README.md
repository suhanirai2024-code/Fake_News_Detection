# 🛡️ NewsGuard AI

## Fake News Detection & Analyzer

NewsGuard AI is a Machine Learning based web application that analyzes a news headline and article content and classifies the news as **Fake** or **Real**.

The application uses **TF-IDF** for text feature extraction and **Logistic Regression** for classification. A Streamlit interface allows users to enter news articles and receive an instant prediction.

---

## 🎯 Objectives

- Detect whether a news article is likely Fake or Real.
- Apply Natural Language Processing techniques to news text.
- Convert textual data into numerical features using TF-IDF.
- Train a Machine Learning classification model.
- Provide an easy-to-use web interface using Streamlit.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib

---

## 📊 Dataset

The dataset contains news articles classified into two categories:

- **Fake News → 0**
- **Real News → 1**

The dataset contains information such as:

- News title
- News article text
- Subject
- Date

After preprocessing and removing duplicate records, the dataset contained **44,689 records**.

---

## 🔄 Methodology

### 1. Data Collection

The dataset consists of Fake and Real news articles.

### 2. Data Preprocessing

The following steps were performed:

- Combined news title and article text.
- Checked for missing values.
- Removed duplicate records.
- Added classification labels.

### 3. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the news text into numerical feature vectors.

A maximum of **5,000 features** was used.

### 4. Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

### 5. Machine Learning Model

A **Logistic Regression** classifier was trained on the TF-IDF features.

### 6. Prediction

The trained model predicts whether the entered news article is:

- 🔴 Fake
- 🟢 Real

The application also displays the prediction probabilities.

---

## 📈 Model Performance

The Logistic Regression model achieved an accuracy of approximately:

**98.95%**

### Confusion Matrix

| Actual / Predicted | Fake | Real |
|---|---:|---:|
| Fake | 4645 | 51 |
| Real | 43 | 4199 |

---

## 🌐 Web Application

The Streamlit application provides:

- News headline input
- News article input
- Fake/Real prediction
- Prediction confidence
- Fake probability
- Real probability
- Article statistics
- Explanation of the ML pipeline

---

## 📁 Project Structure

```text
Fake_News_Detection/
│
├── dataset/
│   ├── Fake.csv
│   ├── True.csv
│   └── cleaned_data.csv
│
├── app.py
├── check_dataset.py
├── prepare_data.py
├── train_model.py
├── model.pkl
├── tfidf.pkl
├── requirements.txt
└── README.md