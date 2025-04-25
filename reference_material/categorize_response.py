"""
This script assigns categories to open-ended responses using GPT-4.
Reads data from a CSV file, filters open-ended questions, sends each response
to GPT-4 for categorization based on a predefined list of categories, and saves the results.
"""

import os
import pandas as pd
import openai

# Ensure the OpenAI API key is set as an environment variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variable.")
openai.api_key = OPENAI_API_KEY

# Define the list of valid categories for classification.
CATEGORIES = [
    'Seek Support', 'Cultural Connection', 'Therapy/Professional Help',
    'Mindfulness & Meditation', 'Exercise', 'Healthy Lifestyle', 'Self-Care Practices',
    'Breathing Techniques', 'Grounding Techniques', 'Routine & Structure',
    'Small Goals', 'Cognitive Behavioral Strategies', 'Limit Stressors', 'Limit Substances', 'Medication',
    'Relaxation Techniques', 'Community Engagement', 'Journaling', 'Engage in Hobbies', 'Distraction'
]

# Input and output file paths.
INPUT_CSV = 'final_prompt_response.csv'
OUTPUT_CSV = 'categorized_responses.csv'


def chat_with_gpt(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def categorize_response(response_text: str) -> str:

    # Build the prompt with instructions and the valid categories.
    prompt = (
        f"Please identify which of the following categories are explicitly or implicitly mentioned in the response. "
        f"Only include categories that are either clearly stated or whose related concepts are strongly suggested. "
        f"Avoid adding categories that are not clearly relevant or only loosely associated. "
        f"Use only the following categories: {', '.join(CATEGORIES)}.\n\n"
        f"Response: \"{response_text}\"\n\n"
        "Return only a comma-separated list of the valid categories from the provided list."
    )
    return chat_with_gpt(prompt)


def main():
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(INPUT_CSV)

    # Filter the DataFrame to include only open-ended questions.
    open_ended = df[df['Question Type'] == 'Open-Ended'].copy()

    # Apply the categorization to each response and store the result in a new column.
    open_ended['Categories'] = open_ended['Response'].apply(categorize_response)

    # Save the updated DataFrame to a CSV file.
    open_ended.to_csv(OUTPUT_CSV, index=False)
    print(f"Categorization completed and saved to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()
