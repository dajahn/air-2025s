import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os

# Set tokenizer parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RerankerDataset(Dataset):
    def _init_(self, queries, documents, labels, tokenizer):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer

    def _len_(self):
        return len(self.queries)

    def _getitem_(self, idx):
        query = self.queries[idx]
        doc = self.documents[idx]
        label = self.labels[idx]

        # For MonoBERT, we concatenate query and document with [SEP]
        text = f"{query} [SEP] {doc}"
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,  # Reduced from 512 to 256
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_training_data(df_collection, df_query, sbert_model, top_k=10):  # Reduced from 20 to 10
    # Prepare document texts
    df_collection["full_text"] = df_collection["title"].fillna('') + " " + df_collection["abstract"].fillna('')
    doc_texts = df_collection["full_text"].tolist()
    doc_uids = df_collection["cord_uid"].tolist()

    # Get embeddings
    doc_embeds = sbert_model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
    query_embeds = sbert_model.encode(df_query["tweet_text"].tolist(), convert_to_tensor=True, show_progress_bar=True)

    # Prepare training data
    train_queries = []
    train_docs = []
    train_labels = []

    for i, qvec in enumerate(query_embeds):
        query = df_query["tweet_text"].iloc[i]
        true_doc_id = df_query["cord_uid"].iloc[i]
        
        # Get top-k candidates
        scores = util.cos_sim(qvec, doc_embeds)[0]
        top_k_indices = torch.topk(scores, k=top_k).indices.tolist()
        
        # Add positive example
        true_doc_idx = doc_uids.index(true_doc_id)
        if true_doc_idx in top_k_indices:
            train_queries.append(query)
            train_docs.append(doc_texts[true_doc_idx])
            train_labels.append(1)
        
        # Add negative examples (only 2 per query to reduce data size)
        neg_count = 0
        for idx in top_k_indices:
            if idx != true_doc_idx and neg_count < 2:  # Only take 2 negative examples
                train_queries.append(query)
                train_docs.append(doc_texts[idx])
                train_labels.append(0)
                neg_count += 1

    return train_queries, train_docs, train_labels

def main():
    # Load data
    print("Loading data...")
    df_collection = pd.read_pickle("subtask4b_collection_data.pkl")
    df_query = pd.read_csv("subtask4b_query_tweets_train.tsv", sep="\t")

    # Load first-stage model
    print("Loading first-stage model...")
    sbert_model = SentenceTransformer("fine-tuned-multi-qa-MiniLM-L6-cos-v1")

    # Prepare training data
    print("Preparing training data...")
    train_queries, train_docs, train_labels = prepare_training_data(
        df_collection, df_query, sbert_model, top_k=10  # Reduced from 20 to 10
    )

    # Initialize tokenizer and model
    print("Initializing MonoBERT model...")
    model_name = "answerdotai/ModernBERT-base"  # Using BERT base as the starting point
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    dataset = RerankerDataset(train_queries, train_docs, train_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Increased batch size

    # Training setup
    optimizer = AdamW(model.parameters(), lr=5e-5)  # Increased learning rate
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    print("Starting training...")
    model.train()
    best_loss = float('inf')
    patience = 2  # Reduced patience
    patience_counter = 0

    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            print("Saving best model...")
            model.save_pretrained("fine-tuned-monobert-reranker")
            tokenizer.save_pretrained("fine-tuned-monobert-reranker")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

if _name_ == "_main_":
    main()