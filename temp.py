import pandas as pd

# Function to filter and save the processed data


def filter_csv_file(file_path):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Ensure the relevant columns exist
        if "Property Name" not in df.columns:
            raise Exception(
                "The CSV file must contain 'Property Name' column.")

        # Filter out rows where Property Name starts with "Archive"
        df_filtered = df[~df['Property Name'].str.startswith("Archive")]

        # Save the filtered DataFrame to a new CSV file
        df_filtered.to_csv(
            "adobe_launch_rules_with_actions_filtered.csv", index=False)
        print("Filtered data saved to adobe_launch_rules_with_actions_filtered.csv")

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    # Modify the file path as needed
    file_path = "adobe_launch_rules_with_actions.csv"
    filter_csv_file(file_path)
