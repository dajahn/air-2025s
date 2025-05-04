import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TraditionalIRModel:
    def __init__(self, collection_path, model_type="bm25"):
        """
        Initialize the IR model with the collection dataset.
        :param collection_path: Path to the collection dataset (CORD-19 metadata).
        :param model_type: Retrieval model type ("bm25" or "tfidf").
        """
        self.collection_path = collection_path
        self.model_type = model_type
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cord_uids = None
        self.corpus = None
        self._load_collection()

    def _load_collection(self):
        """
        Load the collection dataset and prepare the corpus for the selected model.
        """
        df_collection = pd.read_pickle(self.collection_path)
        self.corpus = df_collection[['title', 'abstract']].apply(
            lambda x: f"{x['title']} {x['abstract']}", axis=1
        ).tolist()
        self.cord_uids = df_collection['cord_uid'].tolist()

        if self.model_type == "bm25":
            tokenized_corpus = [doc.split(' ') for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        elif self.model_type == "tfidf":
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)

    def retrieve(self, query, top_k=5):
        """
        Retrieve the top-k documents for a given query using the selected model.
        :param query: The query string.
        :param top_k: Number of top documents to retrieve.
        :return: List of top-k document IDs (cord_uids).
        """
        if self.model_type == "bm25":
            tokenized_query = query.split(' ')
            doc_scores = self.bm25.get_scores(tokenized_query)
            indices = np.argsort(-doc_scores)[:top_k]
        elif self.model_type == "tfidf":
            query_vector = self.tfidf_vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            indices = np.argsort(-cosine_similarities)[:top_k]
        return [self.cord_uids[x] for x in indices]

    def evaluate(self, query_data_path, top_k=5):
        """
        Evaluate the model using Mean Reciprocal Rank (MRR@k).
        :param query_data_path: Path to the query dataset (tweets with references).
        :param top_k: Number of top documents to consider for MRR.
        :return: MRR@k score.
        """
        df_query = pd.read_csv(query_data_path, sep='\t')
        df_query['bm25_topk'] = df_query['tweet_text'].apply(
            lambda x: self.retrieve(x, top_k=top_k)
        )
        df_query['in_topk'] = df_query.apply(
            lambda x: 1 / (x['bm25_topk'].index(x['cord_uid']) + 1)
            if x['cord_uid'] in x['bm25_topk'] else 0, axis=1
        )
        return df_query['in_topk'].mean()

    def generate_predictions(self, query_data_path, output_path, top_k=5):
        """
        Generate predictions and save them to a CSV file.
        :param query_data_path: Path to the query dataset (tweets with references).
        :param output_path: Path to save the predictions CSV file.
        :param top_k: Number of top documents to retrieve.
        """
        df_query = pd.read_csv(query_data_path, sep='\t')
        df_query['preds'] = df_query['tweet_text'].apply(
            lambda x: self.retrieve(x, top_k=top_k)
        )
        df_query[['post_id', 'preds']].to_csv(output_path, index=False)

if __name__ == "__main__":
    # Paths to the datasets
    collection_path = '../default-data/subtask4b_collection_data.pkl'
    query_train_path = '../default-data/subtask4b_query_tweets_train.tsv'
    query_dev_path = '../default-data/subtask4b_query_tweets_dev.tsv'
    output_predictions_path = './out/predictions.csv'

    # Initialize and evaluate the model
    model_type = "bm25"  # Change to "tfidf" for TF-IDF
    model = TraditionalIRModel(collection_path, model_type=model_type)

    print(f"Evaluating on training set with {model_type}...")
    train_mrr = model.evaluate(query_train_path, top_k=5)
    print(f"MRR@5 on training set: {train_mrr}")

    print(f"Evaluating on development set with {model_type}...")
    dev_mrr = model.evaluate(query_dev_path, top_k=5)
    print(f"MRR@5 on development set: {dev_mrr}")

    print("Generating predictions...")
    model.generate_predictions(query_dev_path, output_predictions_path, top_k=5)
    print(f"Predictions for type {model_type} saved to {output_predictions_path}")
    
    model_type = "tfidf"  # Change to "tfidf" for TF-IDF
    model = TraditionalIRModel(collection_path, model_type=model_type)

    print(f"Evaluating on training set with {model_type}...")
    train_mrr = model.evaluate(query_train_path, top_k=5)
    print(f"MRR@5 on training set: {train_mrr}")

    print(f"Evaluating on development set with {model_type}...")
    dev_mrr = model.evaluate(query_dev_path, top_k=5)
    print(f"MRR@5 on development set: {dev_mrr}")

    print("Generating predictions...")
    model.generate_predictions(query_dev_path, output_predictions_path, top_k=5)
    print(f"Predictions for type {model_type} saved to {output_predictions_path}")