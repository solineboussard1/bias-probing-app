"""
Finetune BERT to Predict Attribute
This script will read a csv file with response and target attribute labels.
It will split the data into training and validation sets, tokenize the text using a BERT tokenizer,
and train a BERT model for sequence classification.
It will generate a confusion matrix and classification report for the validation set.
Then it uses SHAP to explain the classifcations and generates bar plots for each class.

python bert_finetune.py --input_csv <path_to_csv> --target_attribute <attribute_name>
"""

import argparse
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,
    pipeline, set_seed
)
import shap
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

class ResponseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    parser = argparse.ArgumentParser(description="BERT Classification for Attribute Prediction")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--target_attribute", type=str, required=True, help="The target attribute to classify")
    args = parser.parse_args()
    
    # Read CSV file
    df = pd.read_csv(args.input_csv)
    
    # For Gender and Race add baseline class to rows with gender and race not specified
    if args.target_attribute in ["Race", "Gender"]:
        df.loc[df['Gender'].isna() & df['Race'].isna(), args.target_attribute] = 'baseline'
        df.dropna(subset=[args.target_attribute], inplace=True)
    else:
        df.dropna(subset=[args.target_attribute], inplace=True)
    
    # Encode target attribute labels
    label_encoder = LabelEncoder()
    df[args.target_attribute] = label_encoder.fit_transform(df[args.target_attribute])
    
    # Split the data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Response'].tolist(),
        df[args.target_attribute].tolist(),
        test_size=0.2,
        random_state=42
    )
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    
    # Create PyTorch datasets
    train_dataset = ResponseDataset(train_encodings, train_labels)
    val_dataset = ResponseDataset(val_encodings, val_labels)
    
    # Load BERT model for sequence classification
    num_labels = len(label_encoder.classes_)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_dir='./logs',
        save_steps=500,
        logging_steps=500,
        load_best_model_at_end=True
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train the model
    trainer.train()
    
    # Pipeline for text classification
    pipeline_model = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        truncation=True 
    )
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(pipeline_model)
    
    # Compute SHAP values for a subset of validation texts
    shap_values = explainer(val_texts[:30])
    
    # Plot SHAP bar plots for each class
    for class_index in range(len(label_encoder.classes_)):
        class_name = label_encoder.inverse_transform([class_index])[0]
        print(f"Top words impacting class {class_name}:")
        shap.plots.bar(shap_values[:, :, class_index].mean(0))
    
    # Make predictions on the validation set
    predictions = trainer.predict(val_dataset)
    val_preds = predictions.predictions.argmax(axis=1)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    cmd = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    cmd.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=75)
    plt.title(f"Confusion Matrix for {args.target_attribute} Prediction")
    plt.show()
    
    # Generate and print classification report
    report = classification_report(val_labels, val_preds, target_names=label_encoder.classes_)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
