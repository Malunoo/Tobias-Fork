try:
    import pandas as pd
    import json
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Specify the JSONL file name directly
    jsonl_file_name = '2023.enriched.jsonl'  # Change 'data.jsonl' to your actual JSONL file name

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
        y = df['is_tech']  # Assume 'is_tech' is the target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split complete.")
    else:
        print("Data split skipped due to missing 'is_tech' column.")

    # Save the preprocessed dataset for future use
    output_file_path = jsonl_file_name.replace('.jsonl', '_preprocessed.csv')
    print(f"Saving the preprocessed dataset to {output_file_path}...")
    df.to_csv(output_file_path, index=False)
    print("Preprocessed dataset saved successfully.")

    print("Preprocessing complete. The script is ready for machine learning modeling.")
except Exception as e:
    print(f"An error occurred: {e}")


##########################################################################################################


# import pandas as pd
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# try:
#     print("Loading data from .jsonl file...")
#     data = []
#     counter = 0
#     text_file_with_path = r'2023.enriched.jsonl'
#     with open(text_file_with_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line))
#             counter += 1
#             if counter % 10000 == 0:
#                 print(f"Processed {counter} lines...")
#     print(f"Data loaded successfully. Total lines processed: {counter}")

#     df = pd.json_normalize(data)
#     print("Data normalized into DataFrame.")
#     print(df.columns)  # Check the actual column names

#     print("Cleaning data...")
#     # Removing duplicates
#     df.drop_duplicates(inplace=True)
#     # Handling missing values for numerical and categorical data
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col].fillna('Unknown', inplace=True)  # For categorical data
#         else:
#             df[col].fillna(df[col].median(), inplace=True)  # For numerical data
#     print("Data cleaned.")

#     print("Extracting features...")
#     swedish_stopwords = stopwords.words('swedish')

#     # List of text columns to potentially include, adjust as per actual DataFrame content
#     text_columns = [
#         'description', 'must_have.skills', 'must_have.languages', 
#         'must_have.work_experiences', 'must_have.education', 'must_have.education_level',
#         'nice_to_have.skills', 'nice_to_have.languages', 'nice_to_have.work_experiences', 
#         'nice_to_have.education', 'nice_to_have.education_level',
#         'description.text', 'description.text_formatted', 'description.company_information',
#         'description.needs', 'description.requirements', 'description.conditions'
#     ]

#     # Filter out the columns that actually exist in the DataFrame
#     existing_text_columns = [col for col in text_columns if col in df.columns]

#     if existing_text_columns:
#         df['combined_text'] = df[existing_text_columns].applymap(str).agg(' '.join, axis=1)
#     else:
#         print("None of the specified text columns are present in the DataFrame.")

#     if 'combined_text' in df.columns:
#         tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words=swedish_stopwords)
#         tfidf_features = tfidf_vectorizer.fit_transform(df['combined_text'])
#         df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#         df = pd.concat([df.drop(existing_text_columns + ['combined_text'], axis=1), df_tfidf], axis=1)
#         print("TF-IDF features extracted and added.")

#     # One-hot encoding categorical variables
#     categorical_columns = [
#         'occupation_field', 'occupation_group', 'employer.name', 
#         'working_hours_type.label', 'remote_work', 'employment_type', 
#         'duration.label'
#     ]
#     existing_categorical_columns = [col for col in categorical_columns if col in df.columns]
#     if existing_categorical_columns:
#         df = pd.get_dummies(df, columns=existing_categorical_columns)
#         print("Features extracted and one-hot encoding done.")

#     print("Splitting data into training and test sets...")
#     if 'is_tech' in df.columns:
#         X = df.drop('is_tech', axis=1)
#         y = df['is_tech']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         print("Data split complete.")
#     else:
#         print("Data split skipped due to missing 'is_tech' column.")

#     output_file_path = text_file_with_path.replace('.jsonl', '_preprocessed.csv')
#     df.to_csv(output_file_path, index=False)
#     print(f"Preprocessed dataset saved successfully at {output_file_path}")

#     print("Preprocessing complete. The script is ready for machine learning modeling.")
# except FileNotFoundError as e:
#     print(f"File not found error: {e}")
# except json.JSONDecodeError as e:
#     print(f"JSON decode error: {e}")
# except MemoryError as e:
#     print(f"Memory error during processing: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
