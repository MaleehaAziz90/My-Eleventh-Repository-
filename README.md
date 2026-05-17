# My-Eleventh-Repository-
Train and Compare Multiple Models (Naive Bayes, SVM, Logistic Regression) for NLP Tasks
# ======================================================
# NLP MODEL COMPARISON PROJECT
# Naive Bayes vs SVM vs Logistic Regression
# ======================================================

# =====================
# IMPORT LIBRARIES
# =====================

import pandas as pd
import numpy as np
import re
import string
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================
# DOWNLOAD NLTK DATA
# =====================

nltk.download('stopwords')
nltk.download('wordnet')

# =====================
# LOAD DATASET
# =====================

print("\nLoading Dataset...")

df = pd.read_csv("dataset.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

# =====================
# TEXT PREPROCESSING
# =====================

print("\nPreprocessing Text...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Join words
    return " ".join(words)

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

print("\nPreprocessing Completed!")

# =====================
# FEATURES & LABELS
# =====================

X = df["clean_text"]
y = df["label"]

# =====================
# TF-IDF VECTORIZATION
# =====================

print("\nApplying TF-IDF Vectorization...")

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Save vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nVectorization Completed!")

# =====================
# TRAIN TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# =====================
# DEFINE MODELS
# =====================

models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Store results
results = []

# Best model tracking
best_model = None
best_accuracy = 0
best_model_name = ""

# =====================
# TRAIN & EVALUATE
# =====================

for name, model in models.items():

    print("\n" + "="*50)
    print(f"TRAINING MODEL: {name}")
    print("="*50)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # =====================
    # METRICS
    # =====================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="positive",
        average="binary"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="positive",
        average="binary"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="positive",
        average="binary"
    )

    # =====================
    # CROSS VALIDATION
    # =====================

    cv_scores = cross_val_score(
        model,
        X_vectorized,
        y,
        cv=5
    )

    # =====================
    # DISPLAY RESULTS
    # =====================

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1-Score : {f1:.2f}")

    print(f"Cross Validation Mean Accuracy: {cv_scores.mean():.2f}")

    # =====================
    # STORE RESULTS
    # =====================

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # =====================
    # SAVE BEST MODEL
    # =====================

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

    # =====================
    # CONFUSION MATRIX
    # =====================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"

    plt.savefig(filename)

    plt.close()

# =====================
# RESULTS TABLE
# =====================

results_df = pd.DataFrame(results)

print("\n" + "="*60)
print("MODEL COMPARISON TABLE")
print("="*60)

print(results_df)

# =====================
# SAVE RESULTS TABLE
# =====================

results_df.to_csv("model_results.csv", index=False)

# =====================
# ACCURACY COMPARISON GRAPH
# =====================

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=results_df
)

plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)

plt.savefig("accuracy_comparison.png")

plt.close()

print("\nAccuracy Comparison Graph Saved!")

# =====================
# GRID SEARCH (BONUS)
# =====================

print("\nPerforming GridSearchCV on Logistic Regression...")

param_grid = {
    'C': [0.1, 1, 10]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best GridSearch Score:", grid.best_score_)

# =====================
# SAVE BEST MODEL
# =====================

joblib.dump(best_model, "best_model.pkl")

print(f"\nBest Model Saved: {best_model_name}")

# =====================
# USER INPUT PREDICTION
# =====================

print("\n" + "="*60)
print("TEXT PREDICTION SYSTEM")
print("="*60)

while True:

    user_text = input("\nEnter text (or type 'exit'): ")

    if user_text.lower() == "exit":
        print("\nExiting Program...")
        break

    # Preprocess input
    clean_input = preprocess_text(user_text)

    # Vectorize input
    input_vector = vectorizer.transform([clean_input])

    print("\nPredictions:\n")

    # Predict using all models
    for name, model in models.items():

        prediction = model.predict(input_vector)[0]

        print(f"{name}: {prediction}")

    print(f"\nBest Model: {best_model_name}")

# =====================
# END OF PROJECT
# =====================

print("\nProject Completed Successfully!")