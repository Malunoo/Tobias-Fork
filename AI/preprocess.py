try:
    import pandas as pd
    import json
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Specify the JSONL file name directly
    jsonl_file_name = '2023.enriched.jsonl'  # Change 'data.jsonl' to your JSONL file name

    # Load and process the .jsonl file line by line to handle large files efficiently
    print("Loading data from .jsonl file...")
    data = []
    counter = 0
    with open(jsonl_file_name, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
            counter += 1
            if counter % 10000 == 0:
                print(f"Processed {counter} lines...")
    print(f"Data loaded successfully. Total lines processed: {counter}")

    # The rest of the code remains the same...
    # Normalizing JSON data into a flat DataFrame
    df = pd.json_normalize(data)
    print("Data normalized into DataFrame.")

    # Continue with data cleaning, feature extraction, and preprocessing as previously coded
    # ...

except Exception as e:
    print(f"An error occurred: {e}")
