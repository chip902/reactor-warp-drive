import json
import pandas as pd
from matplotlib import pyplot as plt

# Load significant functions from JSON file


def load_significant_functions(json_file):
    with open(json_file, "r") as f:
        significant_functions = json.load(f)
    return significant_functions

# Function to visualize significant functions


def visualize_significant_functions(significant_functions):
    # Convert the dictionary to a DataFrame for easier plotting
    df = pd.DataFrame(list(significant_functions.items()),
                      columns=['Function', 'Count'])
    df = df.sort_values(by='Count', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(df['Function'], df['Count'], color='skyblue')
    plt.xlabel('Count')
    plt.ylabel('JavaScript Function')
    plt.title('Significant JavaScript Functions Mentioned')
    plt.gca().invert_yaxis()
    plt.show()

# Main function


def main():
    try:
        significant_functions = load_significant_functions(
            "significant_functions.json")
        visualize_significant_functions(significant_functions)
    except Exception as e:
        print("Error during visualization:", e)


if __name__ == "__main__":
    main()
