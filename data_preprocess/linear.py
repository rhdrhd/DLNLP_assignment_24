import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('essays.csv', encoding='mac_roman')  #Ensure the correct encoding is specified
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

# Train a separate Logistic Regression model for each personality trait
models = {}
scores = {}

for label in label_columns:
    model = LogisticRegression()
    model.fit(X_train_counts, y_train[label])
    models[label] = model
    # Predicting the test set results and calculating accuracy
    y_pred = model.predict(X_test_counts)
    accuracy = accuracy_score(y_test[label], y_pred)
    scores[label] = accuracy

# Output the results
print("Accuracy scores for each personality trait:")
for label in scores:
    print(f"{label}: {scores[label]:.2f}")