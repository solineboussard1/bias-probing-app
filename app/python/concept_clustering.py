import os
import json
import sys
import numpy as np
import requests
from collections import Counter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings(texts, hf_api_key):
    if not hf_api_key:
        raise RuntimeError("Missing Hugging Face API key")
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    resp = requests.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    embs = np.array(resp.json())
    if embs.ndim == 3:
        embs = embs.mean(axis=1)
    return embs

lemmatizer = WordNetLemmatizer()

def normalize_concept(concept: str) -> str:
    tokens = word_tokenize(concept.lower())
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

def get_best_label(concepts, hf_api_key):
    """Find the most representative concept using cosine similarity to centroid."""
    if not concepts:
        return ""
    embeddings = get_embeddings(concepts, hf_api_key)
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    similarities = cosine_similarity(embeddings, centroid).flatten()
    best_index = similarities.argmax()
    return concepts[best_index]

def process_clustering(concept_frequencies, eps=0.25, min_samples=2, min_freq_threshold=3, max_clusters=16, min_clusters=4, hf_api_key=None):
    if not concept_frequencies:
        return [], {}

    # Normalize and expand concepts by frequency
    concepts = []
    for concept, freq in concept_frequencies:
        normalized = normalize_concept(concept)
        concepts.extend([normalized] * freq)

    embeddings = get_embeddings(concepts, hf_api_key)

    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings)

        clusters_dict = {}
        for idx, label in enumerate(labels):
            clusters_dict.setdefault(label, []).append(idx)

        cluster_centroids = {label: np.mean(embeddings[indices], axis=0) for label, indices in clusters_dict.items()}

        small_labels = [label for label, indices in clusters_dict.items() if len(indices) < min_freq_threshold]
        for small_label in small_labels:
            if len(clusters_dict) == 1:
                break
            small_centroid = cluster_centroids[small_label].reshape(1, -1)
            best_label = None
            best_sim = -1
            for label, centroid in cluster_centroids.items():
                if label == small_label:
                    continue
                sim = cosine_similarity(small_centroid, centroid.reshape(1, -1))[0][0]
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            if best_label is not None:
                clusters_dict[best_label].extend(clusters_dict[small_label])
                del clusters_dict[small_label]
                cluster_centroids[best_label] = np.mean(embeddings[clusters_dict[best_label]], axis=0)
                del cluster_centroids[small_label]

        if len(clusters_dict) > max_clusters:
            kmeans = KMeans(n_clusters=max_clusters, random_state=0)
            kmeans_labels = kmeans.fit_predict(embeddings)
            clusters_dict = {}
            for idx, label in enumerate(kmeans_labels):
                clusters_dict.setdefault(label, []).append(idx)
            cluster_centroids = {label: np.mean(embeddings[indices], axis=0) for label, indices in clusters_dict.items()}
        
        if len(clusters_dict) < min_clusters and len(concepts) > min_clusters:
            kmeans = KMeans(n_clusters=min_clusters, random_state=0)
            kmeans_labels = kmeans.fit_predict(embeddings)
            clusters_dict = {}
            for idx, label in enumerate(kmeans_labels):
                clusters_dict.setdefault(label, []).append(idx)
            cluster_centroids = {label: np.mean(embeddings[indices], axis=0) for label, indices in clusters_dict.items()}

        clusters = []
        for label, indices in clusters_dict.items():
            cluster_list = [concepts[i] for i in indices]
            concept_counts = Counter(cluster_list)
            sorted_items = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)

            best_label = get_best_label(cluster_list, hf_api_key)

            clusters.append({
                'id': label,
                'concepts': [item[0] for item in sorted_items],
                'frequency': [item[1] for item in sorted_items],
                'total_frequency': len(cluster_list),
                'label': best_label
            })

        clusters.sort(key=lambda x: x['total_frequency'], reverse=True)
        for i, cluster in enumerate(clusters):
            cluster['id'] = i

        new_concept_to_cluster = {concept: cluster['id'] for cluster in clusters for concept in cluster['concepts']}

        return clusters, new_concept_to_cluster

    except Exception as e:
        print(f"Error during clustering: {e}", file=sys.stderr)
        freq_count = Counter(concepts)
        cluster = {
            'id': 0,
            'concepts': list(freq_count.keys()),
            'frequency': list(freq_count.values()),
            'label': list(freq_count.keys())[0] if freq_count else "",
            'total_frequency': sum(freq_count.values())
        }
        return [cluster], {c: 0 for c in concepts}

def cluster_concepts(input_data):
    hf_api_key = input_data.get("userApiKeys", {}).get("huggingface")
    all_concepts = input_data.get("all", [])
    demographic_data = input_data.get("demographics", {}) or {"baseline": all_concepts}

    all_clusters, concept_to_cluster = process_clustering(all_concepts, hf_api_key=hf_api_key)

    demographic_clusters = {}
    for demo, demo_concepts in demographic_data.items():
        cluster_freq = {cluster['id']: 0 for cluster in all_clusters}
        for concept, freq in demo_concepts:
            normalized = normalize_concept(concept)
            cluster_id = concept_to_cluster.get(normalized, None)
            if cluster_id is not None:
                cluster_freq[cluster_id] += freq

        demo_clusters = []
        for cluster in all_clusters:
            cid = cluster['id']
            demo_clusters.append({
                'id': cid,
                'concepts': cluster['concepts'],
                'frequency': cluster['frequency'],
                'total_frequency': cluster_freq[cid],
                'label': cluster['label']
            })
        demographic_clusters[demo] = demo_clusters

    return {"all": all_clusters, "demographics": demographic_clusters}

if __name__ == "__main__":
    try:
        input_str = sys.stdin.read()
        input_data = json.loads(input_str)
        clusters = cluster_concepts(input_data)
        print(json.dumps(clusters))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
