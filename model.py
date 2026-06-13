import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from data_loader import fetch_and_load_data

MODEL_FILE = "text_classifier_pipeline.joblib"

def train_classification_pipeline():
    """
    Executes data loading, text preprocessing configuration, 
    and model training before saving the final architecture pipeline.
    """
    # 1. Fetch data from data_loader
    df = fetch_and_load_data()
    
    # Dataset structure columns: text_content (features) & label (targets)
    X = df['text_content'].astype(str)
    y = df['label'] # Expected labels: 'AI' vs 'Human' (or numeric equivalents)

    # 2. Split into training and evaluation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training parameters: {len(X_train)} rows | Testing parameters: {len(X_test)} rows.")

    # 3. Create an efficient text modeling pipeline
    # Converts text strings to TF-IDF token matrices, then solves via Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=25000, stop_words='english')),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000))
    ])

    # 4. Fit the ML pipeline
    print("Fitting model to textual parameters...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate the results
    predictions = pipeline.predict(X_test)
    print("\n--- Model Training Metrics ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Summary:")
    print(classification_report(y_test, predictions))

    # 6. Export pipeline asset
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Pipeline model safely stored to disk at: {MODEL_FILE}")
    return pipeline

def get_trained_pipeline():
    """
    Returns the pipeline model asset. Checks if a pre-compiled local file 
    exists; otherwise runs the training compiler dynamically.
    """
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        print("Pre-compiled model asset not identified locally. Compiling new model now...")
        return train_classification_pipeline()

if __name__ == "__main__":
    train_classification_pipeline()