import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import os

# Set tokenizer parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 20  # Number of candidates from first stage
FINAL_K = 5  # Final number of predictions

# Load data
print("Loading data...")
df_collection = pd.read_pickle("subtask4b_collection_data.pkl")
df_query = pd.read_csv("subtask4b_query_tweets_dev.tsv", sep="\t")

# Prepare document texts
df_collection["full_text"] = df_collection["title"].fillna('') + " " + df_collection["abstract"].fillna('')
doc_texts = df_collection["full_text"].tolist()
doc_uids = df_collection["cord_uid"].tolist()

# Load first-stage model (fine-tuned SBERT)
print("Loading first-stage model...")
sbert_model = SentenceTransformer("fine-tuned-multi-qa-MiniLM-L6-cos-v1")
doc_embeds = sbert_model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
query_embeds = sbert_model.encode(df_query["tweet_text"].tolist(), convert_to_tensor=True, show_progress_bar=True)

# Get initial candidates using SBERT
print("Getting initial candidates...")
topk_candidates = {}
for i, qvec in enumerate(query_embeds):
    scores = util.cos_sim(qvec, doc_embeds)[0]
    top_k = torch.topk(scores, k=TOP_K)
    indices = top_k.indices.tolist()
    topk_candidates[df_query["post_id"].iloc[i]] = [(doc_uids[j], doc_texts[j]) for j in indices]

# Load fine-tuned MonoBERT reranker
print("Loading fine-tuned MonoBERT reranker...")
reranker_path = "fine-tuned-monobert-reranker"
tokenizer = AutoTokenizer.from_pretrained(reranker_path)
model = AutoModelForSequenceClassification.from_pretrained(reranker_path).to(DEVICE)
model.eval()

# Rerank candidates
print("Reranking candidates...")
predictions = []

for i, qid in enumerate(tqdm(df_query["post_id"])):
    query = df_query["tweet_text"].iloc[i]
    candidates = topk_candidates[qid]
    
    # Prepare inputs for reranker
    texts = [f"{query} [SEP] {text}" for _, text in candidates]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(DEVICE)
    
    # Get reranker scores
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()
    
    # Sort by reranker scores
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_preds = [doc_id for (doc_id, _), _ in ranked[:FINAL_K]]
    
    predictions.append({
        "post_id": qid,
        "true": df_query["cord_uid"].iloc[i],
        "preds": top_preds
    })

# Calculate MRR@5
def mrr_at_k(predictions, k=5):
    total = 0.0
    for pred in predictions:
        if pred["true"] in pred["preds"]:
            rank = pred["preds"].index(pred["true"]) + 1
            total += 1 / rank
    return total / len(predictions)

mrr_score = mrr_at_k(predictions, k=5)
print(f"MRR@5: {mrr_score:.4f}")

# Save predictions
df_out = pd.DataFrame()
df_out['post_id'] = [p['post_id'] for p in predictions]
df_out['preds'] = [str(p['preds']) for p in predictions]
df_out.to_csv('predictions_monobert_reranker.tsv', sep='\t', index=False)