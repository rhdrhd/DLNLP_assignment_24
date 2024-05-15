import pandas as pd
import openai
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# Initialize the OpenAI client with your API key
openai.api_key = 'sk-proj-EwRsyEt1chy2ZSdlumsRT3BlbkFJqgPaeKVzUDyOawgAUFKH'

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the essays dataset with binary Big Five personality trait scores.
    """
    try:
        df = pd.read_csv(filepath, encoding='mac_roman')
        if 'TEXT' not in df.columns:
            raise ValueError("The required column 'TEXT' is not present in the dataset.")
        return df
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def create_few_shot_prompt(examples: pd.DataFrame, target_essay: str, zero_shot: bool = False) -> str:
    """
    Create a few-shot learning prompt with multiple examples or a zero-shot prompt.

    Parameters:
    - examples: DataFrame containing example essays and their trait scores.
    - target_essay: The essay to be analyzed.
    - zero_shot: Whether to create a zero-shot prompt (no examples).

    Returns:
    - prompt: A formatted string to be used as the prompt for the model.
    """
    prompt = """
    Analyze the following text and determine if the author exhibits the following traits: Extraversion (cEXT), Neuroticism (cNEU), Agreeableness (cAGR), Conscientiousness (cCON), Openness (cOPN). 
    The rules are as follows, for each trait indicate 1 if it is exhibited, 0 if not. 
    Please provide results in JSON format without further explanation and without indicating it is a JSON file.
    """

    if not zero_shot:
        for _, row in examples.iterrows():
            prompt += f"""
            Here is an example:
            Analyze the essay
            "{row['TEXT']}"

            {{
                "cEXT": {1 if row['cEXT'] == 'y' else 0},
                "cNEU": {1 if row['cNEU'] == 'y' else 0}, 
                "cAGR": {1 if row['cAGR'] == 'y' else 0}, 
                "cCON": {1 if row['cCON'] == 'y' else 0}, 
                "cOPN": {1 if row['cOPN'] == 'y' else 0}  
            }}
            """

    prompt += f"""
    Analyze the essay:
    {target_essay}
    """
    
    return prompt

def analyze_personality(dataset: pd.DataFrame, example_count: int = 0) -> Tuple[List[Dict[str, int]], int, pd.DataFrame]:
    """
    Sends essays to ChatGPT and retrieves predicted binary personality traits.
    If an analysis is missed, the corresponding row in `dataset` is dropped.

    Parameters:
    - dataset: DataFrame containing the essays and actual data.
    - example_count: Number of examples to use for few-shot learning. Zero means zero-shot learning.

    Returns:
    - results: List of dictionaries containing the predicted binary personality traits.
    - miss_count: Number of missed analyses.
    - filtered_actual: DataFrame with rows corresponding to missed analyses dropped.
    """
    essays = dataset['TEXT'].tolist()
    actual = dataset[['TEXT', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]
    results = []
    miss_count = 0
    successful_indices = []

    examples = actual.sample(n=example_count) if example_count > 0 else None

    for idx, essay in enumerate(essays):
        prompt = create_few_shot_prompt(examples, essay, zero_shot=(example_count == 0))
        model_name = "gpt-3.5-turbo"

        response = completion_with_backoff(
            model=model_name,
            messages=[
                {"role": "system", "content": "Analyze essays and indicate personality traits."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        response_text = response['choices'][0]['message']['content'].strip()
        trait_scores = parse_json_to_dict(response_text)

        if not trait_scores:
            miss_count += 1
        else:
            results.append(trait_scores)
            successful_indices.append(idx)

    # Filter the actual DataFrame to retain only rows corresponding to successful analyses
    filtered_actual = actual.iloc[successful_indices]

    # Save results to a JSON file
    with open(f'personality_analysis_results_{model_name}.json', 'w') as file:
        json.dump(results, file, indent=4)

    return results, miss_count, filtered_actual

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
        'Trait precision': {k: v for k, v in trait_precision.items()},
        'Trait recall': {k: v for k, v in trait_recall.items()}
    }

def compute_average_metrics(metrics_list):
    # Convert overall accuracy to a DataFrame and compute the mean
    overall_accuracies = pd.DataFrame([metrics['Overall Accuracy'] for metrics in metrics_list])
    average_overall_accuracy = overall_accuracies.mean()[0]

    # Convert trait accuracies, precisions, and recalls to DataFrames and compute the mean
    trait_accuracies = pd.DataFrame([metrics['Trait accuracy'] for metrics in metrics_list])
    average_trait_accuracy = trait_accuracies.mean()

    trait_precisions = pd.DataFrame([metrics['Trait precision'] for metrics in metrics_list])
    average_trait_precision = trait_precisions.mean()

    trait_recalls = pd.DataFrame([metrics['Trait recall'] for metrics in metrics_list])
    average_trait_recall = trait_recalls.mean()

    # Return the averages as a dictionary
    return {
        "Average Overall Accuracy": average_overall_accuracy,
        "Average Trait Accuracy": [item for item in average_trait_accuracy],
        "Average Trait Precision": [item for item in average_trait_precision],
        "Average Trait Recall": [item for item in average_trait_recall]
    }

# Path to your dataset file
#filepath = 'data_preprocess/essays.csv'
filepath = 'essays.csv'
dataset = load_dataset(filepath)
dataset = dataset.sample(frac=0.15, random_state=42)

# Test the analysis for 10 times
all_metrics = []
example_count = 16  # Number of examples for few-shot learning, set to 0 for zero-shot learning

for i in range(3):
    print(f"Iteration {i+1}")
    predicted_traits, miss_count, filtered_actual = analyze_personality(dataset[['TEXT', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']], example_count=example_count)
    print(f"Missed {miss_count} essays")
    metrics = calculate_metrics(filtered_actual, predicted_traits)
    all_metrics.append(metrics)
    #print(f"Iteration {i+1} Metrics: {metrics}")

average_metrics_result = compute_average_metrics(all_metrics)
print(f"Conducting {example_count} shot test, Average Metrics: \n")
for item in average_metrics_result:
    print(f"{item}: {average_metrics_result[item]}") 