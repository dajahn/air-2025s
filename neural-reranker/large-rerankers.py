import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 20
FINAL_K = 5

df_collection = pd.read_pickle("subtask4b_collection_data.pkl")
df_query = pd.read_csv("subtask4b_query_tweets_dev.tsv", sep="\t")

df_collection["full_text"] = df_collection["title"].fillna('') + " " + df_collection["abstract"].fillna('')
doc_texts = df_collection["full_text"].tolist()
doc_uids = df_collection["cord_uid"].tolist()

queries = df_query["tweet_text"].tolist()
query_ids = df_query["post_id"].tolist()
true_labels = df_query["cord_uid"].tolist()
print("Encoding with SBERT for candidate retrieval")
sbert_model = SentenceTransformer("fine-tuned-multi-qa-MiniLM-L6-cos-v1")
doc_embeds = sbert_model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
query_embeds = sbert_model.encode(queries, convert_to_tensor=True, show_progress_bar=True)

topk_candidates = {}
for i, qvec in enumerate(query_embeds):
    scores = util.cos_sim(qvec, doc_embeds)[0]
    top_k = torch.topk(scores, k=TOP_K)
    indices = top_k.indices.tolist()
    topk_candidates[query_ids[i]] = [(doc_uids[j], doc_texts[j]) for j in indices]
#rerankers to compare
reranker_models = {
    "Alibaba-NLP":"Alibaba-NLP/gte-multilingual-reranker-base”, 
    "bge-reranker-v2":"BAAI/bge-reranker-v2-m3",
}

def mrr_at_k(predictions, k=5):
    total = 0.0
    for pred in predictions:
        if pred["true"] in pred["preds"]:
            rank = pred["preds"].index(pred["true"]) + 1
            total += 1 / rank
    return total / len(predictions)

#evaluate reranker
results_summary = {}

for model_name, model_path in reranker_models.items():
    print(f"\nEvaluating Reranker: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)

        predictions = []

        for i, qid in enumerate(tqdm(query_ids)):
            query = queries[i]
            candidates = topk_candidates[qid]
            inputs = tokenizer(
                [query] * len(candidates),
                [text for _, text in candidates],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(DEVICE)

            with torch.no_grad():
                logits = model(**inputs).logits.squeeze().cpu().tolist()

            ranked = sorted(zip(candidates, logits), key=lambda x: x[1], reverse=True)
            top_preds = [doc_id for (doc_id, _), _ in ranked[:FINAL_K]]

            predictions.append({
                "post_id": qid,
                "true": true_labels[i],
                "preds": top_preds
            })

        mrr = mrr_at_k(predictions, k=FINAL_K)

        df_preds = pd.DataFrame([
            {"post_id": pred["post_id"], "preds": pred["preds"]}
            for pred in predictions
        ])

        df_preds.to_csv(f"{model_name}_predictions.tsv", sep="\t", index=False)
        results_summary[model_name] = round(mrr, 4)
        print(f"{model_name} MRR@5: {mrr:.4f}")

    except Exception as e:
        print(f"Error with {model_name}: {e}")

print("\nFinal Reranker MRR@5 Scores:")
print("=" * 35)
for model, score in results_summary.items():
    print(f"{model:<20} MRR@5: {score}")