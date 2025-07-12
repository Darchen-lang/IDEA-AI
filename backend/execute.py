from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import torch

model_path = Path("ideai")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_suggestions(score):
    if score >= 8:
        return "The idea is highly feasible. Consider focusing on initial funding."
    elif score >= 5:
        return "The idea shows potential but requires further refinement."
    else:
        return "The idea may need significant improvements."

def score_idea(idea_text):
    model.eval()  
    inputs = tokenizer(idea_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        output = model(**inputs)
        score = output.logits.item()
        return round(score, 2)

while True:
    new_idea = input("\nEnter your idea (or type 'exit' to quit): ").strip()
    if new_idea.lower() == "exit":
        print("Thank You")
        break
    elif not new_idea:
        print("Please enter a valid idea.")
        continue

    score = score_idea(new_idea)
    print(f"\nFeasibility Score: {score}")
    print("Suggestion:", generate_suggestions(score))
