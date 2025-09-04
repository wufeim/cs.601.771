import json
import pickle

import faiss
import numpy as np


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def load_jsonl(file_path):
    data = {}
    with open(file_path, "r") as f:
        for line in f.read().strip().split("\n"):
            line = json.loads(line)
            data[line["id"]] = {"claim": line["claim"], "evidence": line["evidence"].keys()}
    return data


def load_dataset(data_path):
    train_data = load_jsonl(f"{data_path}/claims_train.jsonl")
    dev_data = load_jsonl(f"{data_path}/claims_dev.jsonl")
    return train_data, dev_data


def compute_map(D, I, evidence_keys):
    ap_list = []
    for i in range(I.shape[0]):
        if len(evidence_keys[i]) == 0:
            continue
        retrieved = I[i, :]

        num_correct = 0
        precision_sum = 0.0
        for j in range(I.shape[1]):
            if str(retrieved[j]) in evidence_keys[i]:
                num_correct += 1
                precision_sum += num_correct / (j + 1)
        ap_list.append(precision_sum / len(evidence_keys[i]))
    return np.mean(ap_list)


def compute_mrr(D, I, evidence_keys):
    mrr_list = []
    for i in range(I.shape[0]):
        if len(evidence_keys[i]) == 0:
            continue
        retrieved = I[i, :]

        first = None
        for j in range(I.shape[1]):
            if str(retrieved[j]) in evidence_keys[i]:
                first = j
                break
        if first is not None:
            mrr_list.append(1 / (first + 1))
        else:
            mrr_list.append(0.0)
    return np.mean(mrr_list)


def main():
    claim_data = load_pickle("scifact_claim_embeddings.pkl")
    evidence_data = load_pickle("scifact_evidence_embeddings.pkl")

    train_data, _ = load_dataset("data/scifact")

    claim_embed = np.array([
        v for k, v in claim_data.items() if k[0] in train_data.keys()
    ], dtype=np.float32)
    evidence_keys = [
        train_data[k[0]]['evidence']
        for k, v in claim_data.items() if k[0] in train_data.keys()]
    evidence_embed = np.array([v for _, v in evidence_data.items()], dtype=np.float32)
    evidence_ids = np.array([k[0] for k in evidence_data.keys()], dtype=np.int64)

    faiss.normalize_L2(claim_embed)
    faiss.normalize_L2(evidence_embed)
    index = faiss.IndexFlatIP(claim_embed.shape[1])
    id_index = faiss.IndexIDMap2(index)
    id_index.add_with_ids(evidence_embed, evidence_ids)

    for k in [1, 10, 50]:
        D, I = id_index.search(claim_embed, k)
        map_k = compute_map(D, I, evidence_keys)
        mrr_k = compute_mrr(D, I, evidence_keys)
        print(f'k={k} map={map_k:.3f} mrr={mrr_k:.3f}')


if __name__ == "__main__":
    main()
