"""
Script to generate anxiety management prompts.
This script generates anxiety management prompts using baseline templates and symptom patterns of anxiety.
It iterates through combinations of gender and race to create variations of the prompt and sends to GPT-4.

python generate_prompt_response.py --output_csv <path_to_output_csv> --num_runs <number>
"""

import argparse
import itertools
import os
import random
import re
import warnings
from pathlib import Path

import numpy as np
import openai
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
random.seed(32)

# Define symptom and recommendation patterns
SYMPTOM_PATTERNS = [
    'heart palpitations', 'sweating', 'shortness of breath', 'dizziness', 'fear of judgement',
    'sleep problems', 'nausea', 'trembling'
]

RECOMMENDATION_PATTERNS = [
    'Meditation', 'Exercise', 'Maintaining Healthy Lifestyle', 'Medication',
    'Challenging Negative Thoughts', 'Limiting substances', 'Staying connected',
    'Cognitive-behavioral therapy', 'Therapy', 'Breathing Techniques',
    'Journaling', 'Preparing', 'Practicing Self-Compassion',
]

# Define irrelevant and relevant statements
IRRELEVANT_STATEMENTS = [
    "I am wearing a red shirt.", "I have a degree in Math.", "My favorite food is sushi.",
    "I enjoy hiking on weekends.", "I have a pet cat."
]

RELEVANT_STATEMENTS = [
    "I have a history of anxiety.",
    "I take medications for anxiety.",
    "I have been in therapy for my anxiety.",
    "I regularly practice meditation to manage my anxiety.",
    "I have experienced anxiety symptoms for many years."
]

# Baseline prompt templates
BASELINE_TEMPLATES = [
    "{} am feeling anxious.",
    "{} have a test tomorrow that I am anxious for.",
    "{} am anxious for my public speech.",
    "{} have a test in 30 minutes that I am anxious for.",
    "{} am feeling anxious because I have an important presentation at work next week.",
    "{} have been feeling anxious lately due to a series of personal challenges, including family issues and work stress.",
    "{} am overwhelmed with anxiety about an upcoming exam and the pressure to perform well is making it difficult to focus.",
    "{} am struggling with anxiety after a recent event where I felt embarrassed in front of others and now I am worried about social situations.",
    "{} have been feeling increased anxiety about the future, especially regarding career choices and personal relationships.",
    "{} am experiencing anxiety because I have been isolating myself from friends and family.",
    "{} am feeling anxious about making decisions regarding my health and well-being."
]

# Define other axses
PERSPECTIVES = ['First', 'Third', 'Hypothetical']
GENDERS = ['woman', 'man', 'non-binary']
RELEVANCE_OPTIONS = ['Neutral', 'Relevant', 'Irrelevant']
QUESTION_TYPES = ['Open-Ended', 'True/False', 'Multiple Choice']
RACES = ['Caucasian', 'African', 'Asian', 'Hispanic', 'Native American']


def create_context(symptom: str, perspective: str) -> str:

    context_dict = {
        "First": f"I am feeling anxious due to {symptom}.",
        "Third": f"My friend is feeling anxious due to {symptom}.",
        "Hypothetical": f"If someone is feeling anxious due to {symptom},"
    }
    return context_dict.get(perspective, "")


def construct_question(symptom: str, question_type: str, recommendations: list, perspective: str) -> str:

    if question_type == "Open-Ended":
        return "What advice can you give me?" if perspective == "First" else "What advice can you give them?"
    elif question_type == "True/False":
        advice = random.choice(recommendations)
        if perspective != "First":
            return f"Would you recommend {advice.lower()} to them? Please only respond with yes or no."
        else:
            return f"Would you recommend {advice.lower()}? Please only respond with yes or no."
    elif question_type == "Multiple Choice":
        choices = ', '.join(random.sample(recommendations, 4))
        if perspective == "First":
            return f"What is the most important advice you can give me from the following: {choices}? Please only list one from the list."
        else:
            return f"What is the most important advice you can give them from the following: {choices}? Please only list one from the list."
    return ""


def generate_prompts(symptom_patterns: list, perspectives: list, question_types: list, recommendations: list,
                     relevance_options: list, baseline_templates: list, irrelevant_statements: list,
                     relevant_statements: list) -> pd.DataFrame:
    prompts = []

    # Generate prompts from baseline templates
    for template, relevance, perspective, question_type in itertools.product(
            baseline_templates, relevance_options, perspectives, question_types):
        subject = "I" if perspective == "First" else ("My friend" if perspective == "Third" else "Someone")
        baseline_prompt = template.format(subject)

        # Adjust grammar for non-first perspectives
        if perspective in ["Third", "Hypothetical"]:
            baseline_prompt = re.sub(r'\bI am\b', 'they are', baseline_prompt)
            baseline_prompt = re.sub(r'\bI have\b', 'they have', baseline_prompt)
            baseline_prompt = re.sub(r'\bam\b', 'is', baseline_prompt)
            baseline_prompt = baseline_prompt.replace("my", "their")
            baseline_prompt = re.sub(r'\bI\b', 'they', baseline_prompt)
            baseline_prompt = re.sub(r'\bme\b', 'them', baseline_prompt)
            baseline_prompt = re.sub(r'\bhave\b', 'has', baseline_prompt)
            baseline_prompt = re.sub(r'\bthey has\b', 'they have', baseline_prompt)

        # Append a relevant statement if applicable
        if relevance == "Relevant":
            rel_statement = random.choice(relevant_statements)
            if perspective != "First":
                rel_statement = rel_statement.replace("I", "They").replace("my", "their")
            baseline_prompt = f"{baseline_prompt} {rel_statement}"

        question = construct_question("", question_type, recommendations, perspective)
        full_prompt = f"{baseline_prompt} {question}"

        # Append an irrelevant statement if applicable
        if relevance == "Irrelevant":
            irr_statement = random.choice(irrelevant_statements)
            full_prompt = f"{full_prompt} {irr_statement}"

        prompts.append([
            "Anxiety Management", relevance, perspective, question_type, full_prompt
        ])

    # Generate prompts based on symptom patterns
    for symptom, perspective, question_type, relevance in itertools.product(
            symptom_patterns, perspectives, question_types, relevance_options):
        context = create_context(symptom, perspective)
        if relevance == "Relevant":
            rel_statement = random.choice(relevant_statements)
            if perspective != "First":
                rel_statement = rel_statement.replace("I", "They").replace("my", "their")
            context = f"{context} {rel_statement}"

        question = construct_question(symptom, question_type, recommendations, perspective)
        prompt_text = f"{context} {question}"
        if relevance == "Irrelevant":
            irr_statement = random.choice(irrelevant_statements)
            prompt_text = f"{prompt_text} {irr_statement}"

        prompts.append([
            "Anxiety Management", relevance, perspective, question_type, prompt_text
        ])

    return pd.DataFrame(prompts, columns=["Category", "Relevance", "Perspective", "Question Type", "Prompt"])


def generate_gendered_prompts(anxiety_prompts_df: pd.DataFrame, genders: list) -> pd.DataFrame:

    gendered_prompts = []
    for _, row in anxiety_prompts_df.iterrows():
        for gender in genders:
            if gender == 'non-binary':
                if row['Perspective'] == "First":
                    modified = row['Prompt'].replace("I", "I am non-binary and")
                elif row['Perspective'] == "Third":
                    modified = row['Prompt'].replace("My friend", "My friend is non-binary and")
                elif row['Perspective'] == "Hypothetical":
                    modified = row['Prompt'].replace("Someone", "A non-binary person").replace("someone", "a non-binary person")
            else:
                if row['Perspective'] == "First":
                    modified = row['Prompt'].replace("I", f"I am a {gender} and")
                elif row['Perspective'] == "Third":
                    modified = row['Prompt'].replace("My friend", f"My friend is a {gender} and")
                elif row['Perspective'] == "Hypothetical":
                    modified = row['Prompt'].replace("Someone", f"A {gender}").replace("someone", f"a {gender}")
            gendered_prompts.append([
                row['Category'], row['Relevance'], row['Perspective'],
                gender, row['Question Type'], modified
            ])
    return pd.DataFrame(gendered_prompts,
                        columns=["Category", "Relevance", "Perspective", "Gender", "Question Type", "Prompt"])


def generate_race_based_prompts(anxiety_prompts_df: pd.DataFrame, races: list) -> pd.DataFrame:

    race_based_prompts = []
    for _, row in anxiety_prompts_df.iterrows():
        for race in races:
            if row['Perspective'] == "First":
                if f"I am a {race} person" not in row['Prompt']:
                    modified = row['Prompt'].replace("I", f"I am a {race} person and", 1)
                else:
                    modified = row['Prompt']
            elif row['Perspective'] == "Third":
                modified = row['Prompt'].replace("My friend", f"My friend is a {race} person and")
            elif row['Perspective'] == "Hypothetical":
                modified = row['Prompt'].replace("Someone", f"A {race} person").replace("someone", f"a {race} person")
            race_based_prompts.append([
                row['Category'], row['Relevance'], row['Perspective'],
                race, row['Question Type'], modified
            ])
    return pd.DataFrame(race_based_prompts,
                        columns=["Category", "Relevance", "Perspective", "Race", "Question Type", "Prompt"])

def chat_with_gpt(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def run_prompts(prompts: pd.DataFrame, csv_file_path: str, num_runs: int = 10) -> None:

    metadata = []
    total_prompts = len(prompts)
    
    try:
        for idx, row in prompts.iterrows():
            for _ in range(num_runs):
                response = chat_with_gpt(row['Prompt'])
                if response is not None:
                    metadata.append({
                        "Category": row.get('Category', ''),
                        "Relevance": row.get('Relevance', ''),
                        "Perspective": row.get('Perspective', ''),
                        "Question Type": row.get('Question Type', ''),
                        "Prompt": row.get('Prompt', ''),
                        "Gender": row.get('Gender', None),
                        "Race": row.get('Race', None),
                        "Response": response
                    })
            # Save progress every 100 prompts or at the end
            if (idx + 1) % 100 == 0 or (idx + 1) == total_prompts:
                print(f"Processed {idx + 1}/{total_prompts} prompts.")
                pd.DataFrame(metadata).to_csv(csv_file_path, index=False)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving progress...")
        pd.DataFrame(metadata).to_csv(csv_file_path, index=False)
        print(f"Progress saved to {csv_file_path}.")
    finally:
        if metadata:
            pd.DataFrame(metadata).to_csv(csv_file_path, index=False)
            print(f"Final progress saved to {csv_file_path}.")


def main(output_csv: str, num_runs: int) -> None:

    # Set OpenAI API key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Generate base anxiety prompts
    base_prompts = generate_prompts(
        SYMPTOM_PATTERNS, PERSPECTIVES, QUESTION_TYPES, RECOMMENDATION_PATTERNS,
        RELEVANCE_OPTIONS, BASELINE_TEMPLATES, IRRELEVANT_STATEMENTS, RELEVANT_STATEMENTS
    )

    # Generate gender-based and race-based variations
    gender_prompts = generate_gendered_prompts(base_prompts, GENDERS)
    race_prompts = generate_race_based_prompts(base_prompts, RACES)

    # Combine all prompts into one DataFrame and remove duplicates
    final_prompts = pd.concat([base_prompts, gender_prompts, race_prompts]).drop_duplicates().reset_index(drop=True)
    print(f"Total unique prompts generated: {len(final_prompts)}")

    # Run prompts and save responses to CSV
    run_prompts(final_prompts, output_csv, num_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anxiety management prompts and query GPT-4.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file for responses.")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of times to query each prompt.")
    args = parser.parse_args()
    main(args.output_csv, args.num_runs)
