import pandas as pd
import openai
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Initialize the OpenAI client with your API key
openai.api_key = 'sk-proj-EwRsyEt1chy2ZSdlumsRT3BlbkFJqgPaeKVzUDyOawgAUFKH'

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the essays dataset with binary Big Five personality trait scores.
    """
    return pd.read_csv(filepath, encoding='mac_roman')

def analyze_personality(essays: List[str]) -> List[Dict[str, int]]:
    """
    Sends essays to ChatGPT and retrieves predicted binary personality traits.
    """
    results = []
    for essay in essays:
        prompt = f"Analyze the following essay and determine if the author exhibits the following traits: Extraversion (cEXT), Neuroticism (cNEU), Agreeableness (cAGR), Conscientiousness (cCON), Openness (cOPN). Indicate presence (1) or absence (0) of each trait:\n{essay}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Analyze essays and indicate personality traits."},
                      {"role": "user", "content": prompt}],
            max_tokens=200
        )
        response_text = response['choices'][0]['message']['content'].strip()
        trait_scores = parse_traits(response_text)
        results.append(trait_scores)
    return results

def parse_traits(response_text: str) -> Dict[str, int]:
    """
    Extracts binary values for personality traits from the model's response.
    """
    traits = {
        "cEXT": "Extraversion",
        "cNEU": "Neuroticism",
        "cAGR": "Agreeableness",
        "cCON": "Conscientiousness",
        "cOPN": "Openness"
    }
    trait_scores = {trait: 0 for trait in traits}
    for trait, name in traits.items():
        # Adjust this logic based on how the model expresses the presence of each trait
        if name in response_text:
            trait_scores[trait] = 1  # Assuming the presence of the trait is explicitly mentioned
    return trait_scores

def calculate_metrics(actual: pd.DataFrame, predicted: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Calculates accuracy, precision, recall, and F1-score for binary trait predictions.
    """
    actual_scores = actual[["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
    predicted_scores = pd.DataFrame(predicted)
    
    # Calculate accuracy, precision, recall, and F1-score for each trait and return the average
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    for trait in actual_scores.columns:
        accuracies.append(accuracy_score(actual_scores[trait], predicted_scores[trait]))
        precisions.append(precision_score(actual_scores[trait], predicted_scores[trait], zero_division=0))
        recalls.append(recall_score(actual_scores[trait], predicted_scores[trait], zero_division=0))
        f1_scores.append(f1_score(actual_scores[trait], predicted_scores[trait], zero_division=0))
    
    return {
        'Average Accuracy': sum(accuracies) / len(accuracies),
        'Average Precision': sum(precisions) / len(precisions),
        'Average Recall': sum(recalls) / len(recalls),
        'Average F1-Score': sum(f1_scores) / len(f1_scores)
    }

# Path to your dataset file
filepath = 'essays.csv'
dataset = load_dataset(filepath)

# Process essays and predict traits
predicted_traits = analyze_personality(dataset['TEXT'].tolist())

# Actual scores dataframe
actual_scores = dataset[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]

# Calculate average accuracy and other metrics
metrics = calculate_metrics(actual_scores, predicted_traits)
print(f"Metrics: {metrics}")