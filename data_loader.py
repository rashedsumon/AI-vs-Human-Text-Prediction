import os
import glob
import pandas as pd
import kagglehub

def fetch_and_load_data():
    """
    Downloads the AI vs Human Text Classification Dataset 2026 
    via kagglehub and returns it as a pandas DataFrame.
    """
    print("Initiating dataset download via kagglehub...")
    # Download latest version of the dataset
    path = kagglehub.dataset_download("alitaqishah/ai-vs-human-text-classification-dataset-2026")
    print("Path to dataset files:", path)
    
    # Locate the target CSV file inside the downloaded directory path
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset package.")
    
    # Read the dataset file (typically 'ai_vs_human_text_2026.csv')
    df = pd.read_csv(csv_files[0])
    return df

if __name__ == "__main__":
    # Test script utility
    try:
        data = fetch_and_load_data()
        print("\nDataset successfully verified and loaded!")
        print(data.head(2))
    except Exception as e:
        print(f"Extraction failed: {e}")