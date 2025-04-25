import pandas as pd
import numpy as np
import json
import sys
import os
import traceback
import math
from scipy.optimize import linear_sum_assignment

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return super().default(obj)

def calculate_raw_agreement(confusion_matrix):
    # Find the optimal matching via Hungarian algorithm.
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    total_matches = confusion_matrix[row_ind, col_ind].sum()
    accuracy = total_matches / np.sum(confusion_matrix)
    return accuracy, row_ind, col_ind

def calculate_contingency_matrices(cluster_labels, topic_labels, pca_cluster_labels,
                                   concept_categories, topic_categories, embedding_categories):
    # Use the original (non–0-indexed) labels preserved as categories for visualization.
    n_clusters = len(concept_categories)
    n_topics = len(topic_categories)
    n_pca_clusters = len(embedding_categories)

    confusion_matrix_cluster_topic = np.zeros((n_clusters, n_topics))
    confusion_matrix_cluster_pca = np.zeros((n_clusters, n_pca_clusters))
    confusion_matrix_topic_pca = np.zeros((n_topics, n_pca_clusters))

    for i in range(len(cluster_labels)):
        confusion_matrix_cluster_topic[cluster_labels[i], topic_labels[i]] += 1
        confusion_matrix_cluster_pca[cluster_labels[i], pca_cluster_labels[i]] += 1
        confusion_matrix_topic_pca[topic_labels[i], pca_cluster_labels[i]] += 1

    # Calculate raw agreement accuracy for each pair.
    accuracy_ct, row_ind_ct, col_ind_ct = calculate_raw_agreement(confusion_matrix_cluster_topic)
    accuracy_cp, row_ind_cp, col_ind_cp = calculate_raw_agreement(confusion_matrix_cluster_pca)
    accuracy_tp, row_ind_tp, col_ind_tp = calculate_raw_agreement(confusion_matrix_topic_pca)

    # Create mapping dictionaries.
    mapping_ct = {int(r): int(c) for r, c in zip(row_ind_ct, col_ind_ct)}
    mapping_cp = {int(r): int(c) for r, c in zip(row_ind_cp, col_ind_cp)}
    mapping_tp = {int(r): int(c) for r, c in zip(row_ind_tp, col_ind_tp)}

    def format_contingency_table(matrix, row_labels, col_labels):
        return {
            "table": matrix.tolist(),
            "rowLabels": row_labels,
            "colLabels": col_labels
        }

    contingency_tables = {
        'cluster_topic': format_contingency_table(confusion_matrix_cluster_topic, concept_categories, topic_categories),
        'cluster_pca': format_contingency_table(confusion_matrix_cluster_pca, concept_categories, embedding_categories),
        'topic_pca': format_contingency_table(confusion_matrix_topic_pca, topic_categories, embedding_categories)
    }

    mapping_data = {
        'cluster_topic_mapping': mapping_ct,
        'cluster_pca_mapping': mapping_cp,
        'topic_pca_mapping': mapping_tp,
        'contingency_tables': contingency_tables,
        'raw_agreement': {
            'cluster_topic': accuracy_ct,
            'cluster_embedding': accuracy_cp,
            'topic_embedding': accuracy_tp
        }
    }
    return mapping_data

def check_agreement(row, cluster_to_topic, cluster_to_pca, topic_to_pca):
    cluster_label = row['Concept_Cluster']
    topic_label = row['Dominant_Topic']
    pca_label = row['Embeddings_Cluster']

    cluster_topic_agree = int(cluster_to_topic.get(cluster_label, -1) == topic_label)
    cluster_pca_agree = int(cluster_to_pca.get(cluster_label, -1) == pca_label)
    topic_pca_agree = int(topic_to_pca.get(topic_label, -1) == pca_label)
    
    return cluster_topic_agree, cluster_pca_agree, topic_pca_agree

def calculate_agreement_scores():
    try:
        csv_path = os.path.join(os.getcwd(), 'public', 'merged_analysis.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        grouped_df = pd.read_csv(csv_path)
        print("Unique values in Concept_Cluster:", grouped_df['Concept_Cluster'].unique(), file=sys.stderr)
        print("Unique values in Dominant_Topic:", grouped_df['Dominant_Topic'].unique(), file=sys.stderr)
        print("Unique values in Embeddings_Cluster:", grouped_df['Embeddings_Cluster'].unique(), file=sys.stderr)
        
        # Drop rows with missing critical labels.
        grouped_df = grouped_df.dropna(subset=['Concept_Cluster', 'Dominant_Topic', 'Embeddings_Cluster'])
        print(grouped_df.isnull().sum(), file=sys.stderr)  # Print count of NaNs

        # Preserve original categories (as strings) for heatmap labels.
        concept_categories = pd.Categorical(grouped_df['Concept_Cluster']).categories.tolist()
        topic_categories = pd.Categorical(grouped_df['Dominant_Topic'].astype(str)).categories.tolist()
        embedding_categories = pd.Categorical(grouped_df['Embeddings_Cluster'].astype(str)).categories.tolist()
        
        # Convert labels to numeric codes based on the preserved order.
        grouped_df['Concept_Cluster'] = pd.Categorical(grouped_df['Concept_Cluster'], categories=concept_categories).codes
        grouped_df['Dominant_Topic'] = pd.Categorical(grouped_df['Dominant_Topic'].astype(str), categories=topic_categories).codes
        grouped_df['Embeddings_Cluster'] = pd.Categorical(grouped_df['Embeddings_Cluster'].astype(str), categories=embedding_categories).codes
        
        # Prepare label arrays.
        cluster_labels = grouped_df['Concept_Cluster'].values
        topic_labels = grouped_df['Dominant_Topic'].values
        pca_cluster_labels = grouped_df['Embeddings_Cluster'].values
        
        # Compute raw agreement scores using Hungarian matching.
        mapping_data = calculate_contingency_matrices(cluster_labels, topic_labels, pca_cluster_labels,
                                                      concept_categories, topic_categories, embedding_categories)
        # Use the raw agreement accuracies from mapping_data.
        agreement_scores = mapping_data['raw_agreement']
        
        # Compute per-row binary agreements for visualization.
        cluster_to_topic = mapping_data['cluster_topic_mapping']
        cluster_to_pca = mapping_data['cluster_pca_mapping']
        topic_to_pca = mapping_data['topic_pca_mapping']
        
        visualization_data = []
        for _, row in grouped_df.iterrows():
            ct_agree, cp_agree, tp_agree = check_agreement(row, cluster_to_topic, cluster_to_pca, topic_to_pca)
            visualization_data.append({
                'pca_one': float(row['PCA_One']) if pd.notnull(row['PCA_One']) else 0.0,
                'pca_two': float(row['PCA_Two']) if pd.notnull(row['PCA_Two']) else 0.0,
                'cluster_topic_agree': ct_agree,
                'cluster_pca_agree': cp_agree,
                'topic_pca_agree': tp_agree
            })

        results = {
            'agreement_scores': agreement_scores,
            'visualization_data': visualization_data,
            'mapping_data': mapping_data
        }
        return results

    except Exception as e:
        error_details = {
            "error": f"Error calculating agreement scores: {str(e)}",
            "traceback": traceback.format_exc(),
            "type": str(type(e).__name__)
        }
        print(json.dumps(error_details), file=sys.stderr)
        raise

if __name__ == "__main__":
    try:
        results = calculate_agreement_scores()
        cleaned_results = json.loads(json.dumps(results, cls=NumpyJSONEncoder))
        print(json.dumps(cleaned_results))
    except Exception as e:
        error_details = {
            "error": f"Failed to calculate agreement scores: {str(e)}",
            "traceback": traceback.format_exc(),
            "type": str(type(e).__name__)
        }
        print(json.dumps(error_details), file=sys.stderr)
        sys.exit(1)

