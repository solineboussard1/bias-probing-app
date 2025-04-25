import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
from sklearn.metrics import silhouette_score

import json
import sys
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(texts: List[str], huggingface_api_key: str) -> np.ndarray:
    """Fetch embeddings from Hugging Face API using a user-provided API key."""
    try:
        if not huggingface_api_key:
            raise ValueError("Missing Hugging Face API key from user input")

        headers = {"Authorization": f"Bearer {huggingface_api_key}"}

        texts = [str(text) for text in texts]

        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        embeddings = np.array(response.json())

        if embeddings.ndim == 3:
            embeddings = np.mean(embeddings, axis=1)

        return embeddings

    except Exception as e:
        logging.error(f"Failed to get embeddings: {str(e)}", exc_info=True)
        raise


def extract_concepts_with_embeddings(input_data: List[Dict[str, Any]], user_api_keys: Dict[str, str]) -> List[Dict[str, Any]]:
    """Extracts concepts by clustering text responses based on Hugging Face embeddings."""
    try:
        logging.info(f"Starting concept extraction with {len(input_data)} items")

        if "huggingface" not in user_api_keys or not user_api_keys["huggingface"]:
            raise ValueError("Missing Hugging Face API key in user input")

        huggingface_api_key = user_api_keys["huggingface"]
        responses = []
        demographics_list = []

        for item in input_data:
            if isinstance(item, dict) and 'response' in item:
                responses.append(item['response'])
                demog = item.get("demographics", {"default": "Unknown"})
                demographics_list.append(demog)

        logging.info(f"Processed {len(responses)} valid responses")

        if len(responses) < 2:
            raise Exception("Need at least 2 responses for clustering")

        embeddings = get_embeddings(responses, huggingface_api_key)

        n_components = min(embeddings.shape[1], len(responses) - 1, 2)
        pca = PCA(n_components=n_components)
        pca_coordinates = pca.fit_transform(embeddings)

        if pca_coordinates.shape[1] == 1:
            pca_coordinates = np.column_stack([pca_coordinates, np.zeros(len(responses))])
        elif pca_coordinates.shape[1] > 2:
            pca_coordinates = pca_coordinates[:, :2]

        if pca_coordinates.shape[1] != 2:
            raise Exception("Failed to generate 2D PCA coordinates")

        explained_variance = pca.explained_variance_ratio_
        print(f"PCA explained variance ratios: {explained_variance}", file=sys.stderr)

        # Find the best number of clusters using Silhouette Score
        n_clusters = 4
        best_score = -1

        for k in range(4, min(len(responses), 18)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)

            if score > best_score:
                best_score = score
                n_clusters = k

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        cluster_concepts = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_responses = np.array(responses)[cluster_mask]
            cluster_embeddings = embeddings[cluster_mask]
            cluster_coordinates = pca_coordinates[cluster_mask]
            cluster_demographics = np.array(demographics_list)[cluster_mask]

            distribution = {}
            for demog in cluster_demographics:
                if isinstance(demog, dict):
                    for category, value in demog.items():
                        if isinstance(value, list):
                            for v in value:
                                key = f"{category}:{v}"
                                distribution[key] = distribution.get(key, 0) + 1
                        else:
                            key = f"{category}:{value}"
                            distribution[key] = distribution.get(key, 0) + 1
                else:
                    key = f"default:{demog}"
                    distribution[key] = distribution.get(key, 0) + 1

            if cluster_coordinates.shape[1] != 2:
                raise Exception(f"Invalid PCA coordinates shape: {cluster_coordinates.shape}")

            cluster_concepts.append({
                "cluster_id": int(i),
                "size": int(np.sum(cluster_mask)),
                "representative_responses": cluster_responses.tolist(),
                "distribution": distribution,
                "embeddings": cluster_embeddings.tolist(),
                "coordinates": cluster_coordinates.tolist()
            })

        logging.info("Successfully completed concept extraction")
        return cluster_concepts

    except Exception as e:
        logging.error(f"Failed to extract concepts: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        logging.info("Reading input from stdin")
        input_data = json.loads(sys.stdin.read())

        if not isinstance(input_data, dict) or "results" not in input_data or "userApiKeys" not in input_data:
            raise ValueError("Invalid input format. Expected keys: 'results' and 'userApiKeys'.")

        results = input_data["results"]
        user_api_keys = input_data["userApiKeys"]

        if not isinstance(user_api_keys, dict):
            raise ValueError("Invalid API keys format. Expected a dictionary.")

        logging.info(f"Received {len(results)} items")

        result = extract_concepts_with_embeddings(results, user_api_keys)

        print(json.dumps(result))
        sys.stdout.flush()
        logging.info("Successfully completed processing")

    except Exception as e:
        logging.error("Fatal error in main", exc_info=True)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
