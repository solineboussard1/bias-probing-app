#!/usr/bin/env python
"""
True/False Response Analysis Script

This script analyzes True/False responses from a GPT response dataset.
It performs the following steps:
  - Loads the CSV file and filters for True/False responses.
  - Classifies responses into 'Yes', 'No', or 'Unable to answer' using regex.
  - Extracts a matched recommendation pattern from the prompt.
  - Runs chi-square tests for each attribute (Gender, Race, Perspective, Relevance)
    comparing the response classification.
  - Generates bar plots (with consistent color palettes) of the percentage distribution 
    for each response type by group.
  - Performs pairwise z-tests for proportions across groups and reports the percentage 
    of significant comparisons.

Usage:
    python tf_analysis.py --input_csv <path_to_csv>

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

# Directory for saving plots
OUTPUT_DIR = "tf_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex pattern for detecting inability to answer
UNABLE_PATTERN = re.compile(
    r"(not be able to provide|can't provide|unable to answer|not qualified|recommend you consult|sorry, but i|i'm not equipped|i recommend consulting)",
    re.IGNORECASE
)

# Recommendation patterns to search for in the prompt
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

ATTRIBUTES = ['Gender', 'Race', 'Perspective', 'Relevance']


def classify_response(response: str) -> str:
    """
    Classify a True/False response into 'Yes', 'No', or 'Unable to answer'.
    
    Args:
        response (str): The raw response text.
        
    Returns:
        str: Classified response.
    """
    response_clean = response.strip().lower()
    if re.search(UNABLE_PATTERN, response_clean):
        return 'Unable to answer'
    if re.search(r'\byes\b', response_clean):
        return 'Yes'
    elif re.search(r'\bno\b', response_clean):
        return 'No'
    else:
        return 'Unable to answer'


def find_pattern(prompt: str) -> str:

    for pattern in RECOMMENDATION_PATTERNS:
        if pattern.lower() in prompt.lower():
            return pattern
    return None



def prepare_data(input_csv: str) -> pd.DataFrame:
    """
    Load and prepare the data for True/False response analysis.
    
    - Loads CSV.
    - Filters for rows where 'Question Type' equals 'True/False'.
    - If both Gender and Race are missing, assigns 'baseline' to Gender.
    - Classifies the response into a new column 'cleaned_response'.
    - Extracts the matched recommendation pattern from the prompt.
    
    Args:
        input_csv (str): Path to the input CSV file.
        
    Returns:
        pd.DataFrame: Prepared DataFrame.
    """
    df = pd.read_csv(input_csv)
    df = df[df['Question Type'] == 'True/False'].copy()
    
    # For rows missing both Gender and Race, assign 'baseline' to Gender
    df.loc[df['Gender'].isna() & df['Race'].isna(), ['Gender', 'Race']] = 'baseline'
    
    # Classify responses and add the new column
    df['cleaned_response'] = df['Response'].apply(classify_response)
    
    # Extract recommendation pattern from prompt
    df['matched_pattern'] = df['Prompt'].apply(find_pattern)
    
    return df


def run_chi_square_tests(df: pd.DataFrame, attribute: str) -> None:

    contingency_table = pd.crosstab(df[attribute], df['cleaned_response'])
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    print(f"Chi-square test for {attribute}:")
    print(f"  Chi-square Statistic: {chi2:.4f}")
    print(f"  P-value: {p:.4f}")
    print(f"  Degrees of Freedom: {dof}\n")


def group_and_calculate_percentages(df: pd.DataFrame, group_attr: str) -> pd.DataFrame:

    grouped = df.groupby(['matched_pattern', group_attr, 'cleaned_response']).size().reset_index(name='count')
    grouped['percent'] = grouped.groupby(['matched_pattern', group_attr])['count'].transform(lambda x: 100 * x / x.sum())
    return grouped


def plot_response_data(grouped: pd.DataFrame, group_attr: str, response_type: str, title: str, palette: dict) -> None:

    data = grouped[grouped['cleaned_response'] == response_type]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='matched_pattern', y='percent', hue=group_attr, palette=palette)
    plt.title(title, fontsize=16)
    plt.xlabel('Recommendation', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.xticks(rotation=35, ha='right', fontsize=14)
    plt.legend(title=group_attr, title_fontsize=14, fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save the plot to the output directory
    filename = os.path.join(OUTPUT_DIR, f"{group_attr}_{response_type}_responses.png")
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


def run_chi_square_by_pattern(grouped: pd.DataFrame, group_attr: str) -> dict:

    results = {}
    for pattern in grouped['matched_pattern'].unique():
        pattern_data = grouped[grouped['matched_pattern'] == pattern]
        contingency_table = pattern_data.pivot_table(index='cleaned_response', columns=group_attr, values='count', fill_value=0)
        chi2, p, dof, _ = chi2_contingency(contingency_table)
        results[pattern] = p
    return results


def z_test_proportions(data: pd.DataFrame, attribute: str, group1: str, group2: str, category: str) -> tuple:

    count_group1 = data[(data[attribute] == group1) & (data['cleaned_response'] == category)].shape[0]
    count_group2 = data[(data[attribute] == group2) & (data['cleaned_response'] == category)].shape[0]
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
    categories = df['cleaned_response'].unique()
    
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



def main():
    parser = argparse.ArgumentParser(description="True/False Response Analysis")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()
    
    # Load and prepare data
    df = prepare_data(args.input_csv)
    
    # Run chi-square tests for each attribute
    for attr in ATTRIBUTES:
        run_chi_square_tests(df, attr)
    
    # For each attribute, group data and generate bar plots for each response type
    for attr in ATTRIBUTES:
        grouped = group_and_calculate_percentages(df, attr)
        palette = COLOR_PALETTES.get(attr, None)
        for response in ['Yes', 'No', 'Unable to answer']:
            plot_response_data(
                grouped,
                group_attr=attr,
                response_type=response,
                title=f'Percentage of {response} Responses by {attr}',
                palette=palette
            )
    
    # Run chi-square tests by recommendation pattern for each attribute
    print("\nChi-square test results by recommendation pattern:")
    for attr in ['Gender', 'Race', 'Perspective', 'Relevance']:
        grouped_attr = group_and_calculate_percentages(df, attr)
        results = run_chi_square_by_pattern(grouped_attr, attr)
        significant_patterns = {pattern: p for pattern, p in results.items() if p < 0.001}
        print(f"\nSignificant patterns for {attr} (p < 0.001):")
        print(significant_patterns)
    
    # Run pairwise z-tests across attributes
    significant_df, all_results_df = run_z_tests(df, ATTRIBUTES)
    print_significant_percentages(all_results_df, significant_df, ATTRIBUTES)

        # … after print_significant_percentages …

    print("\nTop 5 pairwise z‑tests for each attribute (by |z‑statistic|):")
    for attribute in ATTRIBUTES:
        # Filter to this attribute’s full z‑test results
        attr_all = all_results_df[all_results_df['Attribute'] == attribute].copy()
        if attr_all.empty:
            print(f"\nNo z‑tests run for {attribute}")
            continue

        # Compute absolute z‑statistic and pick top 5
        attr_all['abs_stat'] = attr_all['Z-test Statistic'].abs()
        top5 = attr_all.sort_values('abs_stat', ascending=False).head(5)

        # Print them
        print(f"\n⯈ {attribute}")
        print(top5[['Category', 'Group1', 'Group2', 'Z-test Statistic', 'P-value']].to_string(index=False))

  
if __name__ == "__main__":
    main()
