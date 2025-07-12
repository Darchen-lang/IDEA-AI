import pandas as pd
from datasets import Dataset
from transformers import EarlyStoppingCallback
from sklearn.metrics import mean_squared_error

df = pd.read_csv("ideas.csv")
dataset = Dataset.from_pandas(df)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

def tokenize_function(example):
    return tokenizer(example["idea_text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("feasibility_score", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=7,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",  
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    logging_dir="./logs",
    logging_steps=10,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,  
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

eval_results = trainer.evaluate()
print('Assessment:', eval_results)

trainer.save_model("trained_model")
tokenizer.save_pretrained("trained_model")
