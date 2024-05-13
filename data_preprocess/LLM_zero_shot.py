import pandas as pd
import openai
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

# Initialize the OpenAI client with your API key
openai.api_key = 'sk-YourAPIKeyHere' 

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
    miss_count = 0
    for essay in essays:
        prompt = f"""Analyze the following text and determine if the author exhibits the following traits:
        - Extraversion (cEXT)
        - Neuroticism (cNEU)
        - Agreeableness (cAGR)
        - Conscientiousness (cCON)
        - Openness (cOPN)

        Please provide results in JSON format without further explanation and without indicating it is json file. 
        
        The rules are as follows:
            "cEXT": 0,  # 1 if Extraversion is exhibited, 0 if not
            "cNEU": 0,  # 1 if Neuroticism is exhibited, 0 if not
            "cAGR": 0,  # 1 if Agreeableness is exhibited, 0 if not
            "cCON": 0,  # 1 if Conscientiousness is exhibited, 0 if not
            "cOPN": 0   # 1 if Openness is exhibited, 0 if not

        The output should be formated as follows:
        {{
            "cEXT": 0,
            "cNEU": 0, 
            "cAGR": 0, 
            "cCON": 0, 
            "cOPN": 0  
        }}

        Analyze the essay:
        {essay}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Analyze essays and indicate personality traits."},
                      {"role": "user", "content": prompt}],
            max_tokens=200
        )
        response_text = response['choices'][0]['message']['content'].strip()
        trait_scores = parse_json_to_dict(response_text)

        if not trait_scores:
            miss_count += 1
        else:
            results.append(trait_scores)

    # Save results to a JSON file
    with open('personality_analysis_results.json', 'w') as file:
        json.dump(results, file, indent=4)

    return results, miss_count

def parse_json_to_dict(json_str):
    """
    Parses a JSON string and converts it into a Python dictionary.

    Args:
        json_str (str): A JSON-formatted string.

    Returns:
        dict: A Python dictionary representing the JSON data.
    """
    try:
        # Convert the JSON string to a Python dictionary
        result_dict = json.loads(json_str)
        return result_dict
    except json.JSONDecodeError as e:
        # Handle the case where the string is not valid JSON
        print(json_str)
        print("Error parsing JSON:", e)
        return {}

def calculate_metrics(actual: pd.DataFrame, predicted: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Calculates accuracy, precision, recall, and F1-score for binary trait predictions.
    """
    actual_scores = actual[["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]].replace({'y': 1, 'n': 0})
    predicted_scores = pd.DataFrame(predicted)
    
    # Calculate overall accuracy
    actual_scores = actual_scores.reset_index(drop=True)
    correct_map = (actual_scores == predicted_scores)
    correct = correct_map.sum().sum()
    total = correct_map.size
    accuracy = correct / total

    # Calculate accuracy across each trait
    trait_accuracy = correct_map.sum() / correct_map.shape[0]

    # Initialize dictionaries for precision and recall
    trait_precision = {}
    trait_recall = {}

    # Calculate precision and recall for each trait
    for trait in actual_scores.columns:
        true_positives = ((predicted_scores[trait] == 1) & (actual_scores[trait] == 1)).sum()
        false_positives = ((predicted_scores[trait] == 1) & (actual_scores[trait] == 0)).sum()
        false_negatives = ((predicted_scores[trait] == 0) & (actual_scores[trait] == 1)).sum()
        
        # Precision calculation
        if true_positives + false_positives > 0:
            trait_precision[trait] = true_positives / (true_positives + false_positives)
        else:
            trait_precision[trait] = 0
        
        # Sensitivity/recall calculation
        if true_positives + false_negatives > 0:
            trait_recall[trait] = true_positives / (true_positives + false_negatives)
        else:
            trait_recall[trait] = 0

    return {
        'Overall Accuracy': accuracy,
        'Trait accuracy': trait_accuracy,
        'Trait precision': trait_precision,
        'Trait recall': trait_recall
    }

# Path to your dataset file
filepath = 'essays.csv'
dataset = load_dataset(filepath)
dataset = dataset.sample(frac=0.7, random_state=42)
#dataset = dataset.head(20)
# Process essays and predict traits
predicted_traits, miss_count = analyze_personality(dataset['TEXT'].tolist())
print(f"Missed {miss_count} essays")
# Actual scores dataframe
actual_scores = dataset[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]

# Calculate average accuracy and other metrics
metrics = calculate_metrics(actual_scores, predicted_traits)
print(f"Metrics: {metrics}")