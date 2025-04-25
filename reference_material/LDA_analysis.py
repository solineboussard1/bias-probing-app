#!/usr/bin/env python
"""
LDA Analysis Script

This script performs LDA topic modeling on preprocessed text responses and then
applies PCA to the topic distributions. 
It generates the following: 
  - PCA scatter plots and pairplots.
  - Mean topic probability bar plots.
  - ANOVA tests on topic probabilities.
  - KMeans clustering with cluster distribution plots and chi-square tests.
  
python LDA_analysis.py --input_csv <path_to_csv>
"""

import argparse
import os
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.stats import f_oneway, chisquare

# Update global Matplotlib rcParams for consistent styling
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

# Define output directory for plots
output_dir = "lda_plots"
os.makedirs(output_dir, exist_ok=True)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Attributes to run analyses for
ATTRIBUTES = ["Gender", "Race", "Perspective", "Relevance"]

# Define custom color palettes for each attribute
COLOR_PALETTES = {
    'Gender': {
        'baseline': '#1f77b4',
        'man': '#ff7f0e',
        'non-binary': '#2ca02c',
        'woman': '#d62728'
    },
    'Race': {
        'baseline': '#1f77b4',
        'Caucasian': '#ff7f0e',
        'African': '#2ca02c',
        'Asian': '#d62728',
        'Hispanic': '#9467bd',
        'Native American': '#8c564b'
    },
    'Perspective': {
        'First': '#1f77b4',
        'Third': '#ff7f0e',
        'Hypothetical': '#2ca02c'
    },
    'Relevance': {
        'Relevant': '#1f77b4',
        'Neutral': '#ff7f0e',
        'Irrelevant': '#2ca02c'
    }
}

#  Lowercase, remove non-letter characters, lemmatize, and remove stopwords.
def preprocess(text: str) -> str:

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)

#extract top words
def display_topics(model, feature_names, no_top_words=8) -> dict:

    topics = {}
    for idx, topic in enumerate(model.components_):
        topics[f'Topic {idx}'] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics

#run lda analsyis for all attributes
def run_lda_analysis_all_attributes(df: pd.DataFrame, n_topics: int = 12) -> None:

    # Preprocess text for all rows
    df['cleaned_response'] = df['Response'].apply(preprocess)
    df.loc[df['Gender'].isna() & df['Race'].isna(), ['Gender', 'Race']] = 'baseline'
    
    # Pre-train LDA on full data for attributes that use all rows
    vectorizer_full = CountVectorizer(stop_words='english')
    doc_term_matrix_full = vectorizer_full.fit_transform(df['cleaned_response'])
    lda_full = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_full.fit(doc_term_matrix_full)
    full_topic_distributions = lda_full.transform(doc_term_matrix_full)
    topic_cols = [f"Topic {i}" for i in range(n_topics)]
    full_topic_df = pd.DataFrame(full_topic_distributions, columns=topic_cols, index=df.index)
    feature_names_full = vectorizer_full.get_feature_names_out()
    full_topic_words = display_topics(lda_full, feature_names_full, no_top_words=8)
    print("\nTop 5 words per topic (full model):")
    for topic, words in full_topic_words.items():
        print(f"{topic}: {', '.join(words)}")


    
    # Loop over each attribute to perform attribute-specific analyses
    for attribute in ATTRIBUTES:
        print(f"\n--- Analyzing attribute: {attribute} ---")
        df_attr = df.copy()
        
        # For Race and Gender, filter out rows with missing values for that attribute
        if attribute == "Race":
            df_attr = df_attr.dropna(subset=['Race'])
        elif attribute == "Gender":
            df_attr = df_attr.dropna(subset=['Gender'])
        
        # For Gender and Race, train LDA on the filtered data; otherwise, use the full-data LDA
        if attribute in ["Gender", "Race"]:
            vectorizer = CountVectorizer(stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(df_attr['cleaned_response'])
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            feature_names = vectorizer.get_feature_names_out()
            topic_words = display_topics(lda, feature_names, no_top_words=8)
            print(f"\nTop 5 words per topic ({attribute} model):")
            for topic, words in topic_words.items():
                print(f"{topic}: {', '.join(words)}")
            topic_distributions = lda.transform(doc_term_matrix)
            topic_df = pd.DataFrame(topic_distributions, columns=topic_cols, index=df_attr.index)
        else:
            # Use the pre-trained full-data LDA for Perspective and Relevance
            topic_df = full_topic_df.loc[df_attr.index].copy()
        
        # Add the attribute column to the topic DataFrame
        topic_df[attribute] = df_attr[attribute]
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(topic_df[topic_cols])
        topic_df['PCA1'] = pca_results[:, 0]
        topic_df['PCA2'] = pca_results[:, 1]
        
        # Determine dominant topic for annotation
        topic_df['Dominant_Topic'] = topic_df[topic_cols].idxmax(axis=1)
        avg_pca = topic_df.groupby('Dominant_Topic')[['PCA1', 'PCA2']].mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=topic_df, x='PCA1', y='PCA2', hue=attribute,
            s=100, alpha=0.6, edgecolor="w",
            palette=COLOR_PALETTES[attribute]
        )
        for _, row in avg_pca.iterrows():
            plt.text(
                row['PCA1'], row['PCA2'], row["Dominant_Topic"],
                horizontalalignment='center', verticalalignment='center',
                color='black'
            )
        plt.title(f'PCA of LDA Topics Grouped by {attribute}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title=attribute, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        scatter_filename = os.path.join(output_dir, f"PCA_scatter_{attribute}.png")
        plt.savefig(scatter_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved PCA scatter plot to {scatter_filename}")
        
        # PCA Pairplot (4 components)
        pca_4 = PCA(n_components=4)
        pca_4_results = pca_4.fit_transform(topic_df[topic_cols])
        for i in range(4):
            topic_df[f'PCA{i+1}'] = pca_4_results[:, i]
        explained_variance = pca_4.explained_variance_ratio_ * 100
        print("Explained Variance by PCA components:")
        for i, var in enumerate(explained_variance, start=1):
            print(f"PCA{i}: {var:.2f}%")
            
        pairplot = sns.pairplot(
            topic_df, vars=['PCA1', 'PCA2', 'PCA3', 'PCA4'], hue=attribute,
            diag_kind='kde', palette=COLOR_PALETTES[attribute]
        )
        pairplot.fig.suptitle(f'Pairplot of First Four PCA Components by {attribute}', y=1.02)
        pairplot_filename = os.path.join(output_dir, f"PCA_pairplot_{attribute}.png")
        pairplot.fig.savefig(pairplot_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved PCA pairplot to {pairplot_filename}")
        
        # Mean Topic Probability by Group
        if 'Dominant_Topic' in topic_df.columns:
            topic_df.drop(columns='Dominant_Topic', inplace=True)
        mean_topic_prob = topic_df.groupby(attribute).mean().loc[:, topic_cols]
        mean_topic_long = mean_topic_prob.T.reset_index().melt(id_vars='index', var_name='Group', value_name='MeanProbability')
        mean_topic_long.rename(columns={'index': 'Topic'}, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=mean_topic_long, 
            x='Topic', 
            y='MeanProbability', 
            hue='Group',
            palette=COLOR_PALETTES[attribute]
        )
        plt.title(f"Mean Topic Probability by {attribute}")
        plt.xlabel("Topic")
        plt.ylabel("Mean Probability")
        plt.legend(title=attribute)
        plt.tight_layout()
        mean_topic_filename = os.path.join(output_dir, f"Mean_Topic_Probability_{attribute}.png")
        plt.savefig(mean_topic_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved mean topic probability plot to {mean_topic_filename}")

        
        # ANOVA Tests for Topic Probabilities
        anova_results = {}
        unique_groups = topic_df[attribute].unique()
        for topic in topic_cols:
            group_vals = [topic_df[topic_df[attribute] == group][topic] for group in unique_groups]
            if len(unique_groups) > 1:
                f_stat, p_val = f_oneway(*group_vals)
                anova_results[topic] = p_val
            else:
                anova_results[topic] = None
        significant_topics = {topic: p for topic, p in anova_results.items() if p is not None and p < 0.001}
        print(f"\nTopics with significant differences across {attribute} groups (p < 0.001):")
        print(significant_topics)
        
        # KMeans Clustering and Cluster Analysis
        if attribute in ["Gender", "Race"]:
            doc_term_matrix_subset = doc_term_matrix
        else:
            doc_term_matrix_subset = doc_term_matrix_full[df_attr.index, :]
        
        kmeans = KMeans(n_clusters=11, random_state=42)
        cluster_labels = kmeans.fit_predict(doc_term_matrix_subset)
        df_attr = df_attr.copy()
        df_attr['cluster'] = cluster_labels
        
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=df_attr, x='cluster', hue=attribute,
            palette=COLOR_PALETTES[attribute]
        )
        plt.title(f'{attribute} Distribution Across Clusters')
        plt.legend(loc='upper right')
        plt.tight_layout()
        cluster_plot_filename = os.path.join(output_dir, f"Cluster_Distribution_{attribute}.png")
        plt.savefig(cluster_plot_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved cluster distribution plot to {cluster_plot_filename}")
        
        # Chi-square goodness-of-fit tests for clusters
        print(f"\nChi-square goodness-of-fit tests for clusters by {attribute}:")
        for cluster in sorted(df_attr['cluster'].unique()):
            group_counts = df_attr[df_attr['cluster'] == cluster][attribute].value_counts()
            chi2, p = chisquare(group_counts)
            if p < 0.001:
                print(f"Cluster {cluster}: p-value = {p:.4f}")

def main():
    parser = argparse.ArgumentParser(description="LDA Analysis on GPT Responses for All Attributes")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    run_lda_analysis_all_attributes(df, n_topics=12)

if __name__ == "__main__":
    main()
