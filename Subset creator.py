import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def flatten_json(y):
    """ Flatten nested JSON data into a flat dictionary. """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

try:
    print("Reading data from .jsonl file...")
    file_path = os.path.join(os.path.dirname(__file__), "2023.enriched.jsonl")

    if not os.path.exists(file_path):
        print(f"Error: No file found at {file_path}")
        exit()

    print(f".jsonl file path read successfully: {file_path}")

    print("Loading data from .jsonl file...")
    data = []
    counter = 0
    subset_size = 1000  # Adjust this number to change the subset size
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            flat_data = flatten_json(json_data)
            data.append(flat_data)
            counter += 1
            if counter % 1000 == 0:
                print(f"Processed {counter} lines...")
            if counter >= subset_size:
                break
    print(f"Data loaded successfully. Total lines processed: {counter}")

    df = pd.DataFrame(data)

    print("Data normalized into DataFrame.")

    # Analyzing null values
    print("Analyzing null values in columns...")
    null_percentage = df.isnull().mean() * 100
    print("Percentage of null values per column:")
    print(null_percentage)

    # Set a threshold for dropping columns with too many null values, e.g., 50%
    threshold = 50
    columns_to_drop = null_percentage[null_percentage > threshold].index
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"Columns with more than {threshold}% null values dropped.")

    print("Remaining columns:")
    print(df.columns)

    print("Cleaning data...")
    df.drop_duplicates(inplace=True)
    print("Data cleaned.")

    print("Extracting features...")
    swedish_stopwords = stopwords.words('swedish')
    text_columns = ['description_text']  # Update as per your actual text column names
    existing_text_columns = [col for col in text_columns if col in df.columns]
    df['combined_text'] = df[existing_text_columns].fillna('').agg(' '.join, axis=1)

    tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words=swedish_stopwords)
    tfidf_features = tfidf_vectorizer.fit_transform(df['combined_text'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_feature_names)
    df = pd.concat([df.drop(existing_text_columns + ['combined_text'], axis=1), df_tfidf], axis=1)
    print("TF-IDF features extracted and added.")

    print("Splitting data into training and test sets...")
    if 'is_tech' in df.columns:
        X = df.drop('is_tech', axis=1)
        y = df['is_tech']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split complete.")
    else:
        print("Data split skipped due to missing 'is_tech' column.")

    output_file_path = file_path.replace('.jsonl', '_preprocessed_subset.csv')
    df.to_csv(output_file_path, index=False)
    print(f"Preprocessed dataset saved successfully at {output_file_path}")

    print("Preprocessing complete. The script is ready for machine learning modeling.")
except FileNotFoundError as e:
    print(f"File not found error: {e}")
except json.JSONDecodeError as e:
    print(f"JSON decode error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
