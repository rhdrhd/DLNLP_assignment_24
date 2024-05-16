import pandas as pd
import openai
import re
from typing import List, Dict, Tuple
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Retry logic for API calls
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the essays dataset with binary Big Five personality trait scores.

    Parameters:
    - filepath: Path to the CSV file containing the dataset.

    Returns:
    - DataFrame with the dataset.
    """
    try:
        df = pd.read_csv(filepath, encoding='mac_roman')
        if 'TEXT' not in df.columns:
            raise ValueError("The required column 'TEXT' is not present in the dataset.")
        return df
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in the given text for the specified model.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def create_few_shot_prompt(examples: pd.DataFrame, target_essay: str, zero_shot: bool = False, model: str = "gpt-3.5-turbo") -> str:
    """
    Create a few-shot learning prompt with multiple examples or a zero-shot prompt, ensuring the token count does not exceed the model's context length.

    Parameters:
    - examples: DataFrame containing example essays and their trait scores.
    - target_essay: The essay to be analyzed.
    - zero_shot: Whether to create a zero-shot prompt (no examples).
    - model: The model to be used, default is "gpt-3.5-turbo".

    Returns:
    - prompt: A formatted string to be used as the prompt for the model.
    """
    base_prompt = """
    Analyze the following text and determine if the author exhibits the following traits: Extraversion (cEXT), Neuroticism (cNEU), Agreeableness (cAGR), Conscientiousness (cCON), Openness (cOPN).
    Indicate 1 if the trait is exhibited, 0 if not. Use cEXT, cNEU, cAGR, cCON and cOPN to represent each trait.
    Return only results in JSON format without indicating it is JSON file. 
    """

    target_text = f"""
    Analyze the essay:
    {target_essay}
    """

    max_tokens = 16000  # Adjust based on model's token limit

    if zero_shot:
        return base_prompt + target_text

    # Start with a base prompt and add examples one by one until we exceed the token limit
    current_prompt = base_prompt
    for _, row in examples.iterrows():
        example_text = f"""
        Example:
        "{row['TEXT']}"

        {{
            "cEXT": {1 if row['cEXT'] == 'y' else 0},
            "cNEU": {1 if row['cNEU'] == 'y' else 0}, 
            "cAGR": {1 if row['cAGR'] == 'y' else 0}, 
            "cCON": {1 if row['cCON'] == 'y' else 0}, 
            "cOPN": {1 if row['cOPN'] == 'y' else 0}  
        }}
        """
        if count_tokens(current_prompt + example_text + target_text, model) <= max_tokens:
            current_prompt += example_text
        else:
            return {}

    current_prompt += target_text
    return current_prompt

def analyze_personality(dataset: pd.DataFrame, model_name: str, example_count: int = 0) -> Tuple[List[Dict[str, int]], int, pd.DataFrame]:
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
    count = 0
    for idx, essay in enumerate(essays):
        print(f"Now Processing {count}")
        prompt = {}
        while not prompt:
            prompt = create_few_shot_prompt(examples, essay, zero_shot=(example_count == 0))

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
        
        count +=1

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
        # Use regular expression to extract JSON object from the response
        json_match = re.search(r'\{[\s\S]*\}', json_str)
        if json_match:
            json_str_clean = json_match.group(0)
            # Convert the JSON string to a Python dictionary
            result_dict = json.loads(json_str_clean)
            return result_dict
        else:
            print(json_str)
            print("No valid JSON found in the response.")
            return {}
    except json.JSONDecodeError as e:
        # Handle the case where the string is not valid JSON
        print(json_str)
        print("Error parsing JSON:", e)
        return {}

def calculate_metrics(actual: pd.DataFrame, predicted: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Calculates accuracy, precision, recall, and F1-score for binary trait predictions.

    Parameters:
    - actual: DataFrame with actual binary trait scores.
    - predicted: List of dictionaries with predicted binary trait scores.

    Returns:
    - Dictionary with calculated metrics.
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
    """
    Computes average metrics from a list of metric dictionaries.

    Parameters:
    - metrics_list: List of dictionaries with metrics from multiple runs.

    Returns:
    - Dictionary with averaged metrics.
    """
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
        "Average Trait Accuracy": average_trait_accuracy.to_dict(),
        "Average Trait Precision": average_trait_precision.to_dict(),
        "Average Trait Recall": average_trait_recall.to_dict()
    }

def test_chatgpt(fraction, example_count, iterations,api_key):
    """
    Main function to test ChatGPT for personality trait analysis on a dataset.

    Parameters:
    - fraction: Fraction of the dataset to use for testing.
    - example_count: Number of examples for few-shot learning (0 for zero-shot).

    Returns:
    - Prints the average metrics over multiple runs.
    """
    # Initialize the OpenAI client with your API key
    openai.api_key = api_key


    filepath = 'data_preprocess/essays.csv'
    dataset = load_dataset(filepath)
    dataset = dataset.sample(frac=fraction, random_state=42)

    # Test the analysis for 10 times
    all_metrics = []
    example_count = example_count  # Number of examples for few-shot learning, set to 0 for zero-shot learning

    model_name = "gpt-3.5-turbo"
    for i in range(iterations):
        print(f"Iteration {i+1}")
        predicted_traits, miss_count, filtered_actual = analyze_personality(dataset[['TEXT', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']], 
                                                                            model_name=model_name, example_count=example_count)
        print(f"Missed {miss_count} essays")
        metrics = calculate_metrics(filtered_actual, predicted_traits)
        all_metrics.append(metrics)

    average_metrics_result = compute_average_metrics(all_metrics)
    print(f"### Conducting model {model_name} {example_count}-shot test, Average Metrics:")
    for item in average_metrics_result:
        print(f"{item}: {average_metrics_result[item]}")


