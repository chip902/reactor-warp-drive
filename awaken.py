from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import json
from tqdm import tqdm

# Function to compute TF-IDF and find important terms


def extract_significant_terms(texts):
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    # Sum the TF-IDF values for each word
    terms = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    term_scores = {terms[i]: scores[i] for i in range(len(terms))}

    # Sort by importance (descending)
    sorted_terms = dict(
        sorted(term_scores.items(), key=lambda item: item[1], reverse=True))

    # Filter out terms with low importance (optional threshold)
    significant_terms = {k: v for k, v in sorted_terms.items() if v > 1}

    return significant_terms


def main():
    try:
        # Load the CSV file
        df = pd.read_csv("adobe_launch_rules_with_actions.csv")

        # Ensure the relevant columns exist
        if "Action Settings" not in df.columns:
            raise Exception(
                "The CSV file must contain 'Action Settings' column.")

        # Extract text data for NLP analysis
        action_settings = df["Action Settings"].dropna().tolist()

        # Perform AI-driven analysis using TF-IDF
        significant_terms = extract_significant_terms(action_settings)
        print("Significant terms found:", significant_terms)

        # Save significant terms to a JSON file
        with open("significant_js_terms.json", "w") as f:
            json.dump(significant_terms, f, indent=4)

    except Exception as e:
        print("Error during processing:", e)


if __name__ == "__main__":
    main()
