from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

from sklearn.cluster import KMeans
import os

def create_sentence_embeddings(col_name, df):
  title_list = df[col_name].values.tolist()
  #title_list= title_list[:5000]
  sentences = title_list
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  sentence_embeddings = model.encode(sentences)
  return sentences, sentence_embeddings 

for file_name in os.listdir('amazon_data'):
	print(file_name)
	df = pd.read_csv(os.path.join('amazon_data', file_name),names= ['description', 'title', 'feature', 'rank', 'price'])
	sentences, embeddings = create_sentence_embeddings("title", df)

	num_clusters = int(df.shape[0]/100)
	clustering_model = KMeans(n_clusters=num_clusters)
	clustering_model.fit(embeddings)
	cluster_assignment = clustering_model.labels_

	clustered_sentences = [[] for i in range(num_clusters)]
	df_cluster = [0] * df.shape[0]
	for sentence_id, cluster_id in enumerate(cluster_assignment):
		df_cluster[sentence_id] = cluster_id
	df['cluster'] = df_cluster
	df.to_csv(file_name)

