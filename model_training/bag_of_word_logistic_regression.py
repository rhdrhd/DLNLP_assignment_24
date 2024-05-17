import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to set seed for reproducibility
def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def test_bag_of_words():
    seed_everything(42)

    # Load the dataset
    data = pd.read_csv('data_preprocess/essays.csv', encoding='mac_roman') 
    df = pd.DataFrame(data)

    # Convert labels from 'y'/'n' to 1/0
    label_columns = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
    for col in label_columns:
        df[col] = df[col].apply(lambda x: 1 if x == 'y' else 0)

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(df['TEXT'], df[label_columns], test_size=0.2, random_state=42)

    # Creating a bag of words model
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    # Scale the data
    scaler = StandardScaler(with_mean=False)  # Use with_mean=False for sparse data compatibility
    X_train_scaled = scaler.fit_transform(X_train_counts)
    X_test_scaled = scaler.transform(X_test_counts)

    # Train a separate Logistic Regression model for each personality trait
    models = {}
    scores = {}

    # To compute overall accuracy
    total_correct = 0
    total_predictions = 0

    for label in label_columns:
        model = LogisticRegression(max_iter=1000)  # Increase the number of iterations
        model.fit(X_train_scaled, y_train[label])
        models[label] = model
        # Predicting the test set results and calculating accuracy
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test[label], y_pred)
        scores[label] = accuracy
        
        # Calculate overall accuracy
        total_correct += (y_pred == y_test[label]).sum()
        total_predictions += len(y_pred)

    print("### Bag of Words + Linear Regression Model Results:")

    # Overall accuracy
    overall_accuracy = total_correct / total_predictions
    print(f"Overall accuracy across all traits: {overall_accuracy:.4f}")

    # Output the results
    print("Accuracy scores for each personality trait:")
    print(scores)

