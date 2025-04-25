#!/usr/bin/env python
"""
Multiple Choice Response Analysis Script

This script analyzes multiple choice responses from a GPT response dataset.
It performs the following steps:
  - Loads the CSV file and filters for Multiple Choice responses.
  - Normalizes response strings and extracts a canonical recommendation category.
  - Groups data by recommendation pattern and a given attribute, then calculates counts and conditional probabilities.
  - Generates bar plots (using fixed color palettes) for both counts and conditional probabilities.
  - Runs chi-square tests to compare response distributions across groups.
  - Performs pairwise z-tests for proportions and reports the percentage of significant comparisons.
  - Optionally filters significant pairwise comparisons for specific groups and prints the percentages.

Usage:
    python mc_analysis.py --input_csv <path_to_csv>
"""

import argparse
import re
import warnings
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Folder for saving plots
OUTPUT_DIR = "mc_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Recommendation patterns for multiple choice responses
RECOMMENDATION_PATTERNS = [
    'Meditation', 'Exercise', 'Maintaining Healthy Lifestyle', 'Medication',
    'Challenging Negative Thoughts', 'Limiting substances', 'Staying connected',
    'Cognitive-behavioral therapy', 'Therapy', 'Breathing Techniques',
    'Journaling', 'Preparing', 'Practicing Self-Compassion'
]

# Fixed color palettes for consistency across attributes
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

# Attributes to analyze
ATTRIBUTES = ['Gender', 'Race', 'Perspective', 'Relevance']


def prepare_multiple_choice_data(input_csv: str) -> pd.DataFrame:

    df = pd.read_csv(input_csv)
    df = df[df['Question Type'] == 'Multiple Choice'].copy()
    
    # Assign baseline if both Gender and Race are missing
    df.loc[df['Gender'].isna() & df['Race'].isna(), ['Gender', 'Race']] = 'baseline'
    
    # Normalize responses with string replacements
    df['Response'] = df['Response'].str.replace('Maintaining a Healthy Lifestyle', 'Maintaining Healthy Lifestyle', regex=False)
    df['Response'] = df['Response'].str.replace('Maintaining a healthy lifestyle', 'Maintaining Healthy Lifestyle', regex=False)
    df['Response'] = df['Response'].str.replace('Preparation', 'Preparing', regex=False)
    df['Response'] = df['Response'].str.replace('Practice Self-Compassion', 'Practicing Self-Compassion', regex=False)
    df['Response'] = df['Response'].str.replace('Staying Connected', 'Staying connected', regex=False)
    df['Response'] = df['Response'].str.replace(r'\.$', '', regex=True)
    
    # Define a regex pattern to extract a recommendation category (case-insensitive)
    pattern = r'(?i)\b(' + '|'.join(re.escape(cat) for cat in RECOMMENDATION_PATTERNS) + r')\b'
    df['Response'] = df['Response'].apply(lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else 'Invalid')
    
    # Create a mapping from lowercase recommendation to canonical version
    pattern_map = {cat.lower(): cat for cat in RECOMMENDATION_PATTERNS}
    # Compile regex for case-insensitive matching
    compiled_pattern = re.compile(r'\b(' + '|'.join(re.escape(p) for p in RECOMMENDATION_PATTERNS) + r')\b', re.IGNORECASE)
    df['Response'] = df['Response'].apply(lambda x: pattern_map[compiled_pattern.search(x).group(0).lower()] if compiled_pattern.search(x) else 'Unable to Answer')
    
    return df


def group_and_calculate_percentages(df: pd.DataFrame, group_attr: str) -> pd.DataFrame:

    grouped = df.groupby(['Response', group_attr]).size().reset_index(name='count')
    total_counts = grouped.groupby(group_attr)['count'].transform('sum')
    grouped['probability'] = grouped['count'] / total_counts
    return grouped


def plot_conditional_probabilities(grouped: pd.DataFrame, group_attr: str, palette: dict) -> None:

    plt.figure(figsize=(12, 8))
    sns.barplot(data=grouped, x='Response', y='probability', hue=group_attr, palette=palette)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Conditional Probability of Multiple Choice Responses by {group_attr}', fontsize=16)
    plt.xlabel('Recommendation Pattern', fontsize=14)
    plt.ylabel('Conditional Probability', fontsize=14)
    plt.legend(title=group_attr, fontsize=14, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f"{group_attr}_conditional_probabilities.png")
    plt.savefig(filename)
    print(f"Conditional probabilities plot saved to {filename}")
    plt.close()


# Statistical Testing Functions

def run_chi_square_tests(df: pd.DataFrame, attribute: str) -> None:

    contingency_table = pd.crosstab(df[attribute], df['Response'])
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    print(f"Chi-square test for {attribute}:")
    print(f"  Chi-square Statistic: {chi2:.4f}")
    print(f"  P-value: {p:.4f}")
    print(f"  Degrees of Freedom: {dof}\n")


def run_chi_square_by_pattern(grouped: pd.DataFrame, group_attr: str) -> dict:

    results = {}
    for pattern in grouped['Response'].unique():
        pattern_data = grouped[grouped['Response'] == pattern]
        contingency_table = pattern_data.pivot_table(index='Response', columns=group_attr, values='count', fill_value=0)
        chi2, p, dof, _ = chi2_contingency(contingency_table)
        results[pattern] = p
    return results


def z_test_proportions(data: pd.DataFrame, attribute: str, group1: str, group2: str, category: str) -> tuple:

    count_group1 = data[(data[attribute] == group1) & (data['Response'] == category)].shape[0]
    count_group2 = data[(data[attribute] == group2) & (data['Response'] == category)].shape[0]
    total_group1 = data[data[attribute] == group1].shape[0]
    total_group2 = data[data[attribute] == group2].shape[0]
    
    count = np.array([count_group1, count_group2])
    nobs = np.array([total_group1, total_group2])
    
    if np.any(nobs == 0):
        return None, None
    
    stat, p_value = proportions_ztest(count, nobs)
    return stat, p_value


def run_z_tests(df: pd.DataFrame, attributes: list) -> (pd.DataFrame, pd.DataFrame):

    significant_results = []
    all_results = []
    categories = df['Response'].unique()
    
    for attribute in attributes:
        groups = df[attribute].unique()
        for category in categories:
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    stat, p_value = z_test_proportions(df, attribute, groups[i], groups[j], category)
                    if stat is not None and p_value is not None:
                        result = {
                            'Attribute': attribute,
                            'Category': category,
                            'Group1': groups[i],
                            'Group2': groups[j],
                            'Z-test Statistic': stat,
                            'P-value': p_value
                        }
                        all_results.append(result)
                        if p_value < 0.001:
                            significant_results.append(result)
    
    significant_df = pd.DataFrame(significant_results).sort_values(by='P-value')
    all_results_df = pd.DataFrame(all_results)
    return significant_df, all_results_df


def print_significant_percentages(all_results_df: pd.DataFrame, significant_df: pd.DataFrame, attributes: list) -> None:
 
    percentage_significant = {}
    for attribute in attributes:
        total = len(all_results_df[all_results_df['Attribute'] == attribute])
        significant = len(significant_df[significant_df['Attribute'] == attribute])
        percentage = (significant / total * 100) if total > 0 else 0
        percentage_significant[attribute] = percentage
    
    print("Percentage of significant pairwise comparisons by attribute (p < 0.001):")
    for attr, perc in percentage_significant.items():
        print(f"{attr}: {perc:.2f}% significant findings")


# -------------------------------
# Main Function & CLI Interface
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multiple Choice Response Analysis")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()
    
    # Load and prepare data
    df = prepare_multiple_choice_data(args.input_csv)
    print("Total multiple choice responses:", df.shape[0])
    
    # For each attribute, group data and generate bar plots for counts and conditional probabilities
    for attr in ATTRIBUTES:
        palette = COLOR_PALETTES.get(attr, None)
        # Plot conditional probabilities
        grouped_prob = group_and_calculate_percentages(df, attr)
        plot_conditional_probabilities(grouped_prob, attr, palette)
    
    # Run chi-square tests for each attribute
    for attr in ATTRIBUTES:
        run_chi_square_tests(df, attr)
    
    # Run chi-square tests by recommendation pattern for each attribute
    print("\nChi-square test results by recommendation pattern:")
    for attr in ATTRIBUTES:
        grouped_attr = group_and_calculate_percentages(df, attr)
        results = run_chi_square_by_pattern(grouped_attr, attr)
        significant_patterns = {pattern: p for pattern, p in results.items() if p < 0.001}
        print(f"\nSignificant patterns for {attr} (p < 0.001):")
        print(significant_patterns)
    
    # Run pairwise z-tests across attributes
    significant_df, all_results_df = run_z_tests(df, ATTRIBUTES)
    print_significant_percentages(all_results_df, significant_df, ATTRIBUTES)
    
    # Optionally, filter significant results for specific groups (example for Gender and Race)
    gender_filter = significant_df[(significant_df['Attribute'] == 'Gender') & 
                                   ((significant_df['Group1'] == 'non-binary') | (significant_df['Group2'] == 'non-binary'))]
    race_filter = significant_df[(significant_df['Attribute'] == 'Race') & 
                                 ((significant_df['Group1'] == 'Native American') | (significant_df['Group2'] == 'Native American'))]
    
    total_gender_significant = len(significant_df[significant_df['Attribute'] == 'Gender'])
    total_race_significant = len(significant_df[significant_df['Attribute'] == 'Race'])
    
    gender_significant_with_nonbinary = gender_filter.shape[0]
    race_significant_with_native_american = race_filter.shape[0]
    
    percentage_gender = (gender_significant_with_nonbinary / total_gender_significant * 100) if total_gender_significant > 0 else 0
    percentage_race = (race_significant_with_native_american / total_race_significant * 100) if total_race_significant > 0 else 0
    
    print(f"Percentage of significant results involving 'non-binary' (Gender): {percentage_gender:.2f}%")
    print(f"Percentage of significant results involving 'Native American' (Race): {percentage_race:.2f}%")

    def get_top_z_scores_by_attribute(significant_df: pd.DataFrame, top_n: int = 5) -> dict:
        top_z_by_attr = {}
        for attr in significant_df['Attribute'].unique():
            attr_df = significant_df[significant_df['Attribute'] == attr]
            top_df = attr_df.reindex(attr_df['Z-test Statistic'].abs().sort_values(ascending=False).index).head(top_n)
            top_z_by_attr[attr] = top_df
        return top_z_by_attr

    # Get and print top 5 z-scores per attribute
    top_z_scores = get_top_z_scores_by_attribute(significant_df, top_n=5)

    for attr, df in top_z_scores.items():
        print(f"\nTop 5 Z-scores for {attr}:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
