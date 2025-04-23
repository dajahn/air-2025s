import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

class TraditionalIRModel:
    def __init__(self, collection_path):
        """
        Initialize the IR model with the collection dataset.
        :param collection_path: Path to the collection dataset (CORD-19 metadata).
        """
        self.collection_path = collection_path
        self.bm25 = None
        self.cord_uids = None
        self._load_collection()

    def _load_collection(self):
        """
        Load the collection dataset and prepare the BM25 corpus.
        """
        df_collection = pd.read_pickle(self.collection_path)
        corpus = df_collection[['title', 'abstract']].apply(
            lambda x: f"{x['title']} {x['abstract']}", axis=1
        ).tolist()
        self.cord_uids = df_collection['cord_uid'].tolist()
        tokenized_corpus = [doc.split(' ') for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, top_k=5):
        """
        Retrieve the top-k documents for a given query using BM25.
        :param query: The query string.
        :param top_k: Number of top documents to retrieve.
        :return: List of top-k document IDs (cord_uids).
        """
        tokenized_query = query.split(' ')
        doc_scores = self.bm25.get_scores(tokenized_query)
        indices = np.argsort(-doc_scores)[:top_k]
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

if __name__ == "__main__":
    # Paths to the datasets
    collection_path = '../default-data/subtask4b_collection_data.pkl'
    query_train_path = '../default-data/subtask4b_query_tweets_train.tsv'
    query_dev_path = '../default-data/subtask4b_query_tweets_dev.tsv'

    # Initialize and evaluate the model
    model = TraditionalIRModel(collection_path)
    print("Evaluating on training set...")
    train_mrr = model.evaluate(query_train_path, top_k=5)
    print(f"MRR@5 on training set: {train_mrr}")

    print("Evaluating on development set...")
    dev_mrr = model.evaluate(query_dev_path, top_k=5)
    print(f"MRR@5 on development set: {dev_mrr}")