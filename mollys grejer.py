### TAR FÖR MYCKET HÅRDVARA FÖR ATT KÖRAS


# import os
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Download necessary NLTK data for TextBlob and stopwords
# def download_nltk_data():
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('brown')
#     nltk.download('stopwords')

# download_nltk_data()

# # Define custom stopwords
# custom_stopwords = [ ... ]  # Your defined stopwords here

# # Get the list of default stopwords for English and combine with custom stopwords
# default_stopwords = stopwords.words('english')
# all_stopwords = default_stopwords + custom_stopwords

# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())

# # Load data
# file_path = '2023_vasili.csv'
# df = pd.read_csv(file_path, low_memory=False)

# # Define sustainability-related terms and filter for mentions of sustainability
# sustainability_terms = [
#     'sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig', 'clean energy', 'green economy',
#     'sustainable tech', 'technology', 'biodegradable', 'sustainable production', 'green technology', 'carbon neutral', 'circular economy', 'hållbar',
#     'hållbar teknologi', 'hållbar produktion',
#     ]  # Your sustainability-related terms here
# filter_pattern = '|'.join(sustainability_terms)
# df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# # Filter DataFrame for sustainability mentions and create a definitive copy for modifications
# sustainable_df = df[df['mentions_sustainability']].copy()

# # Process text for analysis, integrating all stopwords
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords)
# tfidf_matrix = tfidf_vectorizer.fit_transform(sustainable_df['description.text'])

# # Number of clusters
# n_clusters = 5

# # Apply K-Means Clustering
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(tfidf_matrix)

# # Assign the cluster labels to the sustainable_df DataFrame
# sustainable_df['cluster'] = kmeans.labels_

# # Sentiment Analysis
# print("Analyzing sentiments...")
# sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# # Print out the summary statistics of the sentiment scores
# print("Sentiment Analysis Summary:")
# print(sustainable_df['sentiment'].describe())

# # t-SNE Visualization
# print("Visualizing clusters using t-SNE...")
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
# plt.figure(figsize=(10, 6))
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sustainable_df['cluster'], cmap='viridis', alpha=0.6)
# plt.colorbar()
# plt.title('t-SNE Cluster Visualization')
# plt.xlabel('t-SNE Feature 1')
# plt.ylabel('t-SNE Feature 2')
# plt.show()


#-------------------------------------------------------------------------------------------------------------------

### NY VERSION SOM ANVÄNDER DASK IStäLLEt FÖR PANDAS:

# import os
# import dask.dataframe as dd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns
# from dask_ml.feature_extraction.text import HashingVectorizer
# from dask_ml.cluster import KMeans as DaskKMeans

# # Download necessary NLTK data for TextBlob and stopwords
# def download_nltk_data():
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('brown')
#     nltk.download('stopwords')

# download_nltk_data()

# # Define custom stopwords
# custom_stopwords = [ ... ]  # Your defined stopwords here

# # Get the list of default stopwords for English and combine with custom stopwords
# default_stopwords = stopwords.words('english')
# all_stopwords = default_stopwords + custom_stopwords

# # Print the current working directory and change it if necessary
# print("Current Working Directory:", os.getcwd())

# # Load data using Dask
# file_path = '2023_vasili.csv'
# df = dd.read_csv(file_path, low_memory=False, assume_missing=True, error_bad_lines=False, warn_bad_lines=True)
# #df = dd.read_csv(file_path, low_memory=False, on_bad_lines='skip')


# # Define sustainability-related terms and filter for mentions of sustainability
# sustainability_terms = [
#     'sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig', 'clean energy', 'green economy',
#     'sustainable tech', 'technology', 'biodegradable', 'sustainable production', 'green technology', 'carbon neutral', 'circular economy', 'hållbar',
#     'hållbar teknologi', 'hållbar produktion',
#     ]
# filter_pattern = '|'.join(sustainability_terms)
# df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# # Filter DataFrame for sustainability mentions and create a definitive copy for modifications
# sustainable_df = df[df['mentions_sustainability']].compute()

# # Process text for analysis, integrating all stopwords
# vectorizer = HashingVectorizer(n_features=2**10, stop_words=all_stopwords)
# tfidf_matrix = vectorizer.fit_transform(sustainable_df['description.text'])

# # Number of clusters
# n_clusters = 5

# # Apply K-Means Clustering using Dask's KMeans
# kmeans = DaskKMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(tfidf_matrix)

# # Assign the cluster labels to the sustainable_df DataFrame
# sustainable_df['cluster'] = kmeans.labels_.compute()

# # Sentiment Analysis
# print("Analyzing sentiments...")
# sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity, meta=('x', float))

# # Print out the summary statistics of the sentiment scores
# print("Sentiment Analysis Summary:")
# print(sustainable_df['sentiment'].describe().compute())

# # t-SNE Visualization - note that TSNE is very resource-intensive and may not run well on all hardware
# print("Visualizing clusters using t-SNE...")
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(tfidf_matrix.toarray())  # Convert to dense array for TSNE, may need optimization for very large data
# plt.figure(figsize=(10, 6))
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sustainable_df['cluster'], cmap='viridis', alpha=0.6)
# plt.colorbar()
# plt.title('t-SNE Cluster Visualization')
# plt.xlabel('t-SNE Feature 1')
# plt.ylabel('t-SNE Feature 2')
# plt.show()

#---------------------------------------------------------------------------------------------------------------

import os
import dask.dataframe as dd
import numpy as np
import nltk
from nltk.corpus import stopwords
from dask_ml.feature_extraction.text import HashingVectorizer
from dask_ml.cluster import KMeans as DaskKMeans
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data for TextBlob and stopwords
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('brown')
    nltk.download('stopwords')

download_nltk_data()

# Define custom stopwords
custom_stopwords = [ ... ]  # Your defined stopwords here

# Get the list of default stopwords for English and combine with custom stopwords
default_stopwords = stopwords.words('english')
all_stopwords = default_stopwords + custom_stopwords

# Print the current working directory and change it if necessary
print("Current Working Directory:", os.getcwd())

# Load data using Dask
file_path = '2023_vasili.csv'
# Use on_bad_lines='skip' to skip problematic lines
df = dd.read_csv(file_path, low_memory=False, assume_missing=True, on_bad_lines='skip')

# Define sustainability-related terms and filter for mentions of sustainability
sustainability_terms = [
    'sustainability', 'sustainable', 'green', 'environmental', 'renewable', 'hållbarhet', 'miljövänlig', 'clean energy', 'green economy',
    'sustainable tech', 'technology', 'biodegradable', 'sustainable production', 'green technology', 'carbon neutral', 'circular economy', 'hållbar',
    'hållbar teknologi', 'hållbar produktion',
]
filter_pattern = '|'.join(sustainability_terms)
df['mentions_sustainability'] = df['description.text'].str.contains(filter_pattern, case=False, na=False)

# Filter DataFrame for sustainability mentions and create a definitive copy for modifications
sustainable_df = df[df['mentions_sustainability']].compute()

# Process text for analysis, integrating all stopwords
vectorizer = HashingVectorizer(n_features=2**10, stop_words=all_stopwords)
tfidf_matrix = vectorizer.fit_transform(sustainable_df['description.text'])

# Number of clusters
n_clusters = 5

# Apply K-Means Clustering using Dask's KMeans
kmeans = DaskKMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Assign the cluster labels to the sustainable_df DataFrame
sustainable_df['cluster'] = kmeans.labels_.compute()

# Sentiment Analysis
print("Analyzing sentiments...")
sustainable_df['sentiment'] = sustainable_df['description.text'].apply(lambda text: TextBlob(text).sentiment.polarity, meta=('x', float))

# Print out the summary statistics of the sentiment scores
print("Sentiment Analysis Summary:")
print(sustainable_df['sentiment'].describe().compute())

# t-SNE Visualization - note that TSNE is very resource-intensive and may not run well on all hardware
print("Visualizing clusters using t-SNE...")
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(tfidf_matrix.toarray())  # Convert to dense array for TSNE, may need optimization for very large data
plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sustainable_df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar()
plt.title('t-SNE Cluster Visualization')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()
