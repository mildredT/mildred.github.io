import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np

# Load the dataset
df = pd.read_csv('cleaned_dataset.csv')

# Tokenize and extract features using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_tfidf = tfidf_vectorizer.fit_transform(df['Text'])

# Train SVM model with probability=True
svm_model = SVC(kernel='linear', probability=True)  # Enable probability estimation
svm_model.fit(X_tfidf, df['Label'])

# Function to classify text and provide percentage likelihood
def classify_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction_prob = svm_model.predict_proba(text_tfidf)[0]
    classes = svm_model.classes_
    probabilities = dict(zip(classes, prediction_prob))
    return probabilities

# User input text
input_text = input("Enter the text you want to classify: ")

# Classify input text
result = classify_text(input_text)

# Display results
print("\nClassification Results:")
for label, probability in result.items():
    print(f"{label}: {probability * 100:.2f}%")
