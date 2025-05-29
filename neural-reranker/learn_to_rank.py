import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import os
from collections import Counter

# Set tokenizer parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_features(df_collection, df_query, sbert_model, bm25, top_k=200):
    # Get document texts
    doc_texts = df_collection["full_text"].tolist()
    doc_uids = df_collection["cord_uid"].tolist()

    # Get SBERT embeddings
    print("Encoding with SBERT")
    doc_embeds = sbert_model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
    query_embeds = sbert_model.encode(df_query["tweet_text"].tolist(), convert_to_tensor=True, show_progress_bar=True)

    # Prepare features
    features = []
    labels = []
    query_ids = []
    doc_ids = []

    print("Generating features")
    for i, qvec in enumerate(tqdm(query_embeds)):
        query = df_query["tweet_text"].iloc[i]
        true_doc_id = df_query["cord_uid"].iloc[i]

        # Get SBERT scores for all documents
        sbert_scores = util.cos_sim(qvec, doc_embeds)[0].cpu().numpy()

        # Get BM25 scores
        bm25_scores = bm25.get_scores(query.split())

        # Get top-k candidates based on SBERT scores
        top_k_indices = np.argsort(sbert_scores)[-top_k:][::-1]

        # Combine features for top-k candidates
        for idx in top_k_indices:
            doc_text = doc_texts[idx]

            # Basic text features
            query_words = set(query.lower().split())
            doc_words = set(doc_text.lower().split())

            # Word overlap features
            word_overlap = len(query_words.intersection(doc_words))
            word_overlap_ratio = word_overlap / len(query_words) if len(query_words) > 0 else 0

            # Length features
            query_length = len(query.split())
            doc_length = len(doc_text.split())
            length_ratio = min(query_length, doc_length) / max(query_length, doc_length)

            # Combine all features
            feature_vector = [
                sbert_scores[idx],  # SBERT similarity
                bm25_scores[idx],   # BM25 score
                word_overlap,       # Number of overlapping words
                word_overlap_ratio, # Ratio of overlapping words
                query_length,       # Query length
                doc_length,         # Document length
                length_ratio,       # Length ratio
            ]

            features.append(feature_vector)
            labels.append(1 if doc_uids[idx] == true_doc_id else 0)
            query_ids.append(df_query["post_id"].iloc[i])
            doc_ids.append(doc_uids[idx])

    return np.array(features), np.array(labels), query_ids, doc_ids

def get_group_sizes(query_ids):
    return [count for _, count in sorted(Counter(query_ids).items())]

def train_ltr_model(X, y, query_ids, params=None):
    """Train XGBoost LTR model"""
    if params is None:
        params = {
            'objective': 'rank:pairwise',
            'learning_rate': 0.01,  # Further reduced learning rate
            'max_depth': 8,         # Increased depth
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'random_state': 42,
            'eval_metric': ['map@5', 'ndcg@5'],  # Changed to MAP@5 for better MRR alignment
            'max_leaves': 128,      # Increased max leaves
            'grow_policy': 'lossguide',
            'scale_pos_weight': 10  # Added to handle class imbalance
        }

    # Compute group sizes
    train_group_sizes = get_group_sizes(query_ids)

    # Prepare DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(train_group_sizes)

    # Train model with more rounds and better early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,  # Increased number of rounds
        evals=[(dtrain, 'train')],
        early_stopping_rounds=50,  # Increased early stopping rounds
        verbose_eval=True
    )

    # Print feature importance
    importance = model.get_score(importance_type='gain')
    print("\nFeature Importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feat}: {imp:.4f}")

    return model

def main():
    # Load data
    print("Loading data...")
    df_collection = pd.read_pickle("subtask4b_collection_data.pkl")
    df_query = pd.read_csv("subtask4b_query_tweets_train.tsv", sep="\t")

    # Prepare document texts
    print("Preparing document texts...")
    df_collection["full_text"] = df_collection["title"].fillna('') + " " + df_collection["abstract"].fillna('')

    # Initialize models
    print("Initializing models...")
    sbert_model = SentenceTransformer("/content/drive/MyDrive/Colab Notebooks/fine-tuned-multi-qa-MiniLM-L6-cos-v1")
    tokenized_corpus = [doc.split() for doc in df_collection["full_text"]]
    bm25 = BM25Okapi(tokenized_corpus)

    # Prepare features
    X, y, query_ids, doc_ids = prepare_features(
        df_collection,
        df_query,
        sbert_model,
        bm25,
        top_k=200  # Increased top_k for better coverage
    )

    # Convert query_ids to numpy array for indexing
    query_ids = np.array(query_ids)

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split data while preserving query groups
    unique_queries = np.unique(query_ids)
    train_queries = np.random.choice(unique_queries, size=int(0.8 * len(unique_queries)), replace=False)
    train_mask = np.isin(query_ids, train_queries)

    X_train, y_train, q_train = X[train_mask], y[train_mask], query_ids[train_mask]
    X_val, y_val, q_val = X[~train_mask], y[~train_mask], query_ids[~train_mask]

    # Train model
    print("Training LTR model...")
    model = train_ltr_model(X_train, y_train, q_train)

    # Save model and scaler
    model.save_model("ltr_model.json")
    np.save("feature_scaler.npy", scaler)

    # Evaluate on validation set
    val_group_sizes = get_group_sizes(q_val)
    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(val_group_sizes)
    val_preds = model.predict(dval)

    # Calculate MRR@5
    def mrr_at_k(predictions, labels, query_ids, k=5):
        df = pd.DataFrame({
            'query_id': query_ids,
            'pred': predictions,
            'label': labels
        })

        mrr = 0
        total_queries = 0
        for qid in df['query_id'].unique():
            q_df = df[df['query_id'] == qid].sort_values('pred', ascending=False)
            if q_df['label'].sum() > 0:  # Only count queries with relevant documents
                total_queries += 1
                for i, (_, row) in enumerate(q_df.iterrows()):
                    if row['label'] == 1:
                        mrr += 1 / (i + 1)
                        break
        return mrr / total_queries if total_queries > 0 else 0

    mrr_score = mrr_at_k(val_preds, y_val, q_val)
    print(f"\nValidation MRR@5: {mrr_score:.4f}")

    # Print some statistics
    print("\nValidation Statistics:")
    print(f"Total validation queries: {len(np.unique(q_val))}")
    print(f"Queries with relevant documents: {sum(1 for qid in np.unique(q_val) if y_val[q_val == qid].sum() > 0)}")
    print(f"Average relevant documents per query: {y_val.sum() / len(np.unique(q_val)):.2f}")

if __name__ == "__main__":
    main()


""" Created this python file in case notebook gives error

OUTPUT
-----------
Feature Importance:
f0: 487.0022
f3: 89.3573
f1: 36.4904
f2: 21.7450
f4: 11.5527
f5: 8.4758
f6: 7.7138

Validation MRR@5: 0.7262

Validation Statistics:
Total validation queries: 2571
Queries with relevant documents: 2532
Average relevant documents per query: 0.98 """