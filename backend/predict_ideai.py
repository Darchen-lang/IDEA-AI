import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

class IdeaFeasibilityPredictor:
    def __init__(self, model_path="trained_model"): 
        """
        Initializes the predictor by loading the trained model and tokenizer.
        model_path: Directory where the model and tokenizer are saved.
        """
        self.model_path = Path(model_path) 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=1)
        self.model.eval() 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model and tokenizer loaded from: {self.model_path} on device: {self.device}")

    def predict(self, idea_text: str) -> float:
        """
        Predicts the feasibility score for a given idea text.
        idea_text: The input text for which to predict the score.
        Returns the predicted feasibility score as a float.
        """
        if not isinstance(idea_text, str) or not idea_text.strip():
            raise ValueError("Input idea_text must be a non-empty string.")

        inputs = self.tokenizer(
            idea_text,
            return_tensors="pt", 
            truncation=True,     
            padding="max_length", 
            max_length=128       
        ).to(self.device) #

     
        with torch.no_grad():
            output = self.model(**inputs)
            score = output.logits.item() 

        return round(score, 2) 

    def generate_suggestions(self, score):
        if score >= 8:
            return "The idea is highly feasible. Consider focusing on initial funding."
        elif score >= 5:
            return "The idea shows potential but requires further refinement."
        else:
            return "The idea may need significant improvements."



if __name__ == "__main__":
    print("Testing the IdeaFeasibilityPredictor locally...")
    try:
        
        predictor = IdeaFeasibilityPredictor(model_path="trained_model") 
        test_idea = "This is a brilliant new product idea with huge market potential."
        score = predictor.predict(test_idea)
        suggestion = predictor.generate_suggestions(score)
        print(f"Idea: '{test_idea}'")
        print(f"Predicted Feasibility Score: {score}")
        print(f"Suggestion: {suggestion}")

        print("-" * 30)

        test_idea_boring = "This is a very common idea that already exists everywhere."
        score_boring = predictor.predict(test_idea_boring)
        suggestion_boring = predictor.generate_suggestions(score_boring)
        print(f"Idea: '{test_idea_boring}'")
        print(f"Predicted Feasibility Score: {score_boring}")
        print(f"Suggestion: {suggestion_boring}")

    except Exception as e:
        print(f"An error occurred during local test: {e}")
        print("\nIMPORTANT: Did you run 'python tobetrained.py' first? It creates the 'trained_ideai_model' folder needed here.")