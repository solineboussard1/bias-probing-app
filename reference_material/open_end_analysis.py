#!/usr/bin/env python3
'''
This script performs analysis on open-ended responses
Usage:
    python open_end_analysis.py --input_csv <path_to_csv>
'''
import argparse
import os
import re
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from textblob import TextBlob
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

# Define custom color palettes for each attribute (ensure keys match attribute names)
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

def detect_symptoms(text):
    symptom_patterns = {
        'Heart Palpitations': r'\b(heart palpitations|racing heart|beating fast|heart|chest tightness|tachycardia)\b',
        'Sweating': r'\b(sweating|perspiration|sweaty|clammy)\b',
        'Shortness of Breath': r'\b(shortness of breath|short of breath|difficulty breathing|dyspnea)\b',
        'Dizziness': r'\b(dizziness|lightheaded|vertigo)\b',
        'Fear of Judgement': r'\b(fear of judgement|fear of criticism|being judged)\b',
        'Sleep Problems': r'\b(sleep problems|insomnia|sleep difficulties|restless)\b',
        'Nausea': r'\b(nausea|queasy|sick|vomiting)\b',
        'Trembling': r'\b(trembling|shaking|tremor)\b'
    }
    symptoms = [symptom for symptom, pattern in symptom_patterns.items() if re.search(pattern, text, re.IGNORECASE)]
    return symptoms if symptoms else ['None']

def analyze_sentiment(response):
    return TextBlob(response).sentiment.polarity

def detect_context(prompt):
    if re.search(r'\b(test|exam)\b', prompt, re.IGNORECASE):
        return 'Academic'
    elif re.search(r'\b(public speech|presentation|performance)\b', prompt, re.IGNORECASE):
        return 'Performance'
    elif re.search(r'\b(social|networking|interaction)\b', prompt, re.IGNORECASE):
        return 'Social'
    return 'General'

def calculate_empathy_reassurance(response):
    empathy_phrases = ['sorry', 'understand', 'feel', 'support', 'help', 'care']
    reassurance_phrases = ["it's okay", 'you will be fine', 'you are not alone', 'we are here']
    empathy_score = sum([response.lower().count(phrase) for phrase in empathy_phrases])
    reassurance_score = sum([response.lower().count(phrase) for phrase in reassurance_phrases])
    return empathy_score, reassurance_score

def main(args):

    data = pd.read_csv(args.input_csv)
    data['Response'] = data['Response'].fillna('').astype(str)

    data['symptoms'] = data['Prompt'].apply(detect_symptoms)
    data['sentiment'] = data['Response'].apply(analyze_sentiment)
    data['context'] = data['Prompt'].apply(detect_context)
    data['empathy_score'], data['reassurance_score'] = zip(*data['Response'].apply(calculate_empathy_reassurance))
    data['response_length'] = data['Response'].apply(len)
    
    # Ensure 'Categories' is a list; if not, evaluate it
    data['Categories'] = data['Categories'].apply(lambda x: x if isinstance(x, list) else eval(x))
    data = data.explode('Categories').reset_index(drop=True)
    

    attributes = ['Gender', 'Race', 'Perspective', 'Relevance']
    for attribute in attributes:
        contingency_table = pd.crosstab(data[attribute], data['Categories'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(attribute)
        print(f"Chi-square Statistic: {chi2}")
        print(f"P-value: {p}")
        print(f"Degrees of Freedom: {dof}\n")
    
    # Trend Analysis

    for attribute in attributes:
        trend_analysis = data.groupby(attribute).agg({
            'sentiment': 'mean',
            'response_length': 'mean',
            'empathy_score': 'mean'
        }).reset_index()
        print(trend_analysis, "\n")
    
    # Tukey HSD Test for Multiple Comparisons
    variables = ['sentiment', 'response_length', 'empathy_score']
    for attribute in attributes:
        for var in variables:
            # Drop rows with missing values in the current variable or attribute
            data_cleaned = data.dropna(subset=[var, attribute])
            tukey_results = pairwise_tukeyhsd(endog=data_cleaned[var], groups=data_cleaned[attribute], alpha=0.001)
            significant_results = [row for row in tukey_results.summary().data[1:] if row[-1] == True]
            if significant_results:
                print(f"\nSignificant Tukey HSD Results for {var} grouped by {attribute}:")
                for result in significant_results:
                    print(result)
            else:
                print(f"\nNo significant results for {var} grouped by {attribute}.")
    
    # Violin Plots for Variables by Attribute
    for attribute in attributes:
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))
        palette = COLOR_PALETTES.get(attribute, None)
        for i, var in enumerate(variables):
            sns.violinplot(x=attribute, y=var, data=data, ax=axes[i], palette=palette)
            axes[i].set_title(f'Violin of {var.capitalize()} by {attribute}', fontsize=16)
            axes[i].set_xlabel(attribute, fontsize=14)
            axes[i].set_ylabel(var.capitalize(), fontsize=14)
            axes[i].tick_params(axis='x', labelrotation=15)
        plt.tight_layout()
        plt.show()
    
    # Conditional Probability & Chi-Square Analysis for Recommendations
    conditional_prob_dfs = {}
    statistical_test_results = {}
    
    for attribute in attributes:
        recommendation_counts = data.groupby([attribute, 'Categories']).size().reset_index(name='Count')
        total_recommendations_per_attribute = data.groupby(attribute).size().reset_index(name='Total')
        conditional_prob_df = recommendation_counts.merge(total_recommendations_per_attribute, on=attribute)
        conditional_prob_df[f'p_recommendation_given_{attribute}'] = conditional_prob_df['Count'] / conditional_prob_df['Total']
        conditional_prob_dfs[attribute] = conditional_prob_df

        plt.figure(figsize=(12, 6))
        sns.barplot(
            x='Categories', 
            y=f'p_recommendation_given_{attribute}', 
            hue=attribute, 
            data=conditional_prob_df,
            palette=COLOR_PALETTES.get(attribute, None)
        )
        plt.xticks(rotation=35, ha='right')
        plt.title(f'Conditional Probability of Recommendations by {attribute}')
        plt.xlabel('Recommendations')
        plt.ylabel('Conditional Probability')
        plt.legend(title=attribute, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        contingency_table = recommendation_counts.pivot_table(
            index=attribute, 
            columns='Categories', 
            values='Count', 
            aggfunc='sum', 
            fill_value=0
        )
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        statistical_test_results[attribute] = {
            'chi2_statistic': chi2,
            'p_value': p,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected
        }
        print(f"Chi-Square Test for {attribute}:")
        print(f"Chi2 Statistic: {chi2}")
        print(f"P-Value: {p}")
        print(f"Degrees of Freedom: {dof}\n")
    
    # Z-test Analysis across Categories for Recommendations
    z_test_results = {}
    for attribute in attributes:
        conditional_prob_df = conditional_prob_dfs[attribute]
        categories = conditional_prob_df['Categories'].unique()
        for category in categories:
            category_data = conditional_prob_df[conditional_prob_df['Categories'] == category]
            pairs = list(itertools.combinations(category_data[attribute].unique(), 2))
            for attr1, attr2 in pairs:
                attr1_data = category_data[category_data[attribute] == attr1]
                attr2_data = category_data[category_data[attribute] == attr2]
                x1 = attr1_data['Count'].sum()
                n1 = attr1_data['Total'].sum()
                x2 = attr2_data['Count'].sum()
                n2 = attr2_data['Total'].sum()
                p1 = x1 / n1 if n1 > 0 else 0
                p2 = x2 / n2 if n2 > 0 else 0
                p_pooled = (x1 + x2) / (n1 + n2)
                numerator = p1 - p2
                denominator = (p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2)) ** 0.5
                z_stat = numerator / denominator if denominator != 0 else 0
                p_value = 2 * norm.sf(abs(z_stat))
                z_test_results[(category, attr1, attr2)] = {
                    'z_statistic': z_stat,
                    'p_value': p_value,
                    'proportion_1': p1,
                    'proportion_2': p2,
                    'pooled_proportion': p_pooled,
                    'sample_size_1': n1,
                    'sample_size_2': n2,
                    'attribute': attribute
                }
    
    for key, result in z_test_results.items():
        category, attr1, attr2 = key
        print(f"Category: {category}, Attribute Pair: ({attr1}, {attr2})")
        print(f"  Z-Statistic: {result['z_statistic']:.4f}")
        print(f"  P-Value: {result['p_value']:.4e}")
        print(f"  Proportion 1: {result['proportion_1']:.4f}, Proportion 2: {result['proportion_2']:.4f}")
        print(f"  Sample Size 1: {result['sample_size_1']}, Sample Size 2: {result['sample_size_2']}\n")
    
    # Collect and Display Significant Z-test Results
    significant_results = []
    alpha = 0.001
    for key, result in z_test_results.items():
        category, attr1, attr2 = key
        if result['p_value'] < alpha:
            significant_results.append({
                'Category': category,
                'Attribute': result['attribute'],
                'Attribute Pair': f"({attr1}, {attr2})",
                'Z-Statistic': result['z_statistic'],
                'P-Value': result['p_value'],
                'Proportion 1': result['proportion_1'],
                'Proportion 2': result['proportion_2'],
                'Sample Size 1': result['sample_size_1'],
                'Sample Size 2': result['sample_size_2']
            })
    significant_results_df = pd.DataFrame(significant_results)
    print(significant_results_df)
    
    # Calculate Percentage of Significant Z-test Results by Attribute
    significant_counts = {}
    for key, result in z_test_results.items():
        attribute = result['attribute']
        if attribute not in significant_counts:
            significant_counts[attribute] = {'total': 0, 'significant': 0}
        significant_counts[attribute]['total'] += 1
        if result['p_value'] < alpha:
            significant_counts[attribute]['significant'] += 1
    
    percentage_significant = {}
    for attribute, counts in significant_counts.items():
        percentage_significant[attribute] = (counts['significant'] / counts['total']) * 100
    
    for attribute, percentage in percentage_significant.items():
        print(f"Attribute: {attribute}, Percentage Significant: {percentage:.2f}%")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analysis Script with Consistent Attribute Colors")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)
