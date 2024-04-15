try:
    import pandas as pd
    import json
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Reading JSONL file path from a text file for better project structure and security
    print("Reading the .jsonl file path from the text file...")
    text_file_with_path = r'C:\Users\Brandon\Documents\Repositorys\MPTJB\MPTJB\AI\jsonl_file_path.txt'
    with open(text_file_with_path, 'r', encoding='utf-8') as f:
        file_path = f.readline().strip()
    print(f".jsonl file path read successfully: {file_path}")

    # Load and process the .jsonl file line by line to handle large files efficiently
    print("Loading data from .jsonl file...")
    data = []
    counter = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
            counter += 1
            if counter % 10000 == 0:
                print(f"Processed {counter} lines...")
    print(f"Data loaded successfully. Total lines processed: {counter}")

    # Normalizing JSON data into a flat DataFrame
    df = pd.json_normalize(data)
    print("Data normalized into DataFrame.")

    # Clean the Data
    print("Cleaning data...")
    # Users can add specific data cleaning steps here depending on the dataset
    print("Data cleaned.")

    # Feature Extraction
    print("Extracting features...")
    # Convert 'description.text' to TF-IDF features; replace 'description.text' with your column of interest
    if 'description.text' in df.columns:
        # Joining lists into strings if necessary, adjust column name as needed
        df['description.text'] = df['description.text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_features = tfidf_vectorizer.fit_transform(df['description.text'])
        print("TF-IDF features extracted.")
    else:
        print("'description.text' column not found, skipping TF-IDF feature extraction.")

    # One-hot encode selected categorical variables; adjust column names as needed
    for column in ['occupation_field', 'industry']:
        if column in df.columns:
            if df[column].apply(lambda x: isinstance(x, list)).any():
                print(f"Column '{column}' contains list-type data, considering a different handling approach.")
            else:
                df = pd.get_dummies(df, columns=[column])
                print(f"Categorical variable '{column}' one-hot encoded.")
        else:
            print(f"'{column}' column not found. Skipping one-hot encoding for '{column}'.")

    # Preparing data for machine learning
    print("Splitting data into training and test sets...")
    if 'is_tech' in df.columns:
        X = df.drop('is_tech', axis=1)  # Drop the target variable to isolate features; adjust as necessary
        y = target_variable  # Define your target variable here
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split complete.")
    else:
        print("Data split skipped due to missing 'is_tech' column.")

    # Save the preprocessed dataset for future use
    output_file_path = file_path.replace('.jsonl', '_preprocessed.csv')
    print(f"Saving the preprocessed dataset to {output_file_path}...")
    df.to_csv(output_file_path, index=False)
    print("Preprocessed dataset saved successfully.")

    print("Preprocessing complete. The script is ready for machine learning modeling.")
except Exception as e:
    print(f"An error occurred: {e}")