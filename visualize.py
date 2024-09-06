import json
import pandas as pd
from matplotlib import pyplot as plt


def visualize_significant_functions(json_file):
    with open(json_file, 'r') as f:
        significant_functions = json.load(f)

    df = pd.DataFrame(list(significant_functions.items()),
                      columns=['Function', 'Importance'])
    df = df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(df['Function'], df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('JavaScript Function')
    plt.title('Significant JavaScript Functions Detected via TF-IDF')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    visualize_significant_functions("significant_js_terms.json")
