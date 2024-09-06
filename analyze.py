import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from tqdm import tqdm
import pandas as pd
import json

# Starting dictionary of JavaScript functions/keywords
starting_dictionary = {
    "fbevents.js": 0,
    "gtag": 0,
    "Hotjar": 0,
    "ctrk": 0,
    "yimg": 0,
    "epsilon": 0
}

# Function to perform NLP on text data


def extract_significant_functions(texts, initial_dictionary):
    function_call_pattern = re.compile(r'\b\w+\s*\([^)]*\)')
    token_counts = Counter(initial_dictionary)

    for text in tqdm(texts, desc="Processing text data"):
        function_calls = function_call_pattern.findall(text)
        token_counts.update(function_calls)

    # Filter significant functions
    significant_functions = {token: count for token, count in token_counts.items(
    ) if count > 9}  # Adjust threshold as needed
    return significant_functions

# Main function to load existing CSV and perform NLP


def main():
    try:
        # Load the CSV file
        df = pd.read_csv("adobe_launch_rules_with_actions_orig.csv")

        # Ensure the relevant columns exist
        if "Action Settings" not in df.columns:
            raise Exception(
                "The CSV file must contain 'Action Settings' column.")

        # Extract text data for NLP analysis
        action_settings = df["Action Settings"].dropna().str.lower().tolist()

        # Perform NLP on collected texts and update the dictionary
        significant_functions = extract_significant_functions(
            action_settings, starting_dictionary)
        print("Significant functions found:", significant_functions)

        # Save significant functions to a JSON file
        with open("significant_functions.json", "w") as f:
            json.dump(significant_functions, f, indent=4)

    except Exception as e:
        print("Error during processing:", e)


if __name__ == "__main__":
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    main()
