import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_text(text):
    # Tokenize and encode the input text
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    return input_ids, attention_mask

def classify_text(text):
    input_ids, attention_mask = preprocess_text(text)
    # Pass the input through the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=1)
    return predictions

def get_percentage(predictions):
    # Calculate the percentage of human vs ChatGPT predictions
    human_percentage = predictions[0][0].item() * 100
    chatgpt_percentage = predictions[0][1].item() * 100
    return human_percentage, chatgpt_percentage

# Example usage:
input_text = input("Enter the text to classify: ")
predictions = classify_text(input_text)
human_percentage, chatgpt_percentage = get_percentage(predictions)
print("Human Percentage for Input Text:", human_percentage)
print("ChatGPT-3.5 Percentage for Input Text:", chatgpt_percentage)
