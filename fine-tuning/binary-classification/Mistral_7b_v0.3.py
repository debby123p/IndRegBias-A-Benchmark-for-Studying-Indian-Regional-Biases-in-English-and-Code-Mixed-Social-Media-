import os
import torch
import pandas as pd
import numpy as np
import re
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import gc
import shutil
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "" # Target GPU node

CONFIG = {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.3", # Model ID
    "dataset_path": "", # Path to the Dataset
    "output_root_dir": "", # Output Directory
    "seed": 42,
    "max_seq_length": 512,
    "label_map": {0: "non-regional bias", 1: "regional bias"},
    "num_folds": 5
}

MODEL_PROMPT = """
You are an expert in identifying regional biases in social media comments about Indian states and regions. Your task is to classify whether a comment contains regional biases or not.

Task: Classify the given comment as either "REGIONAL BIAS" (1) or "NON-REGIONAL BIAS" (0).

Definition:
- REGIONAL BIAS (1): Comments that reinforce or propagate biases about Indian states, regions, or their people based on linguistic, cultural, economic, political, or infrastructural aspects. The comments can reflect either positive or negative biases towards specific states or regions.
- NON-REGIONAL BIAS (0): Comments that are neutral or factual without generalisations, or unrelated to regional characteristics.

Step-by-Step Analysis Process:
Step 1: Identify Regional References
Think: Does this comment mention or refer to:
- Specific Indian states (e.g., Bihar, Kerala, Punjab, etc.)
- Regional groups (e.g., South Indians, North Indians, etc.)
- Cities or regions within India
- Language communities within India

Step 2: Check for Elements reinforcing biases
Look for these patterns:
- Generalisations about people from a state or a regional group ("All X are Y")
- Assumptions about state/regional characteristics
- Comparative statements that imply superiority/inferiority
- Overgeneralized cultural, linguistic, economic, political, or infrastructural claims

Step 3: Assess the Nature of the Statement
Consider:
- Is this a factual observation or a generalised assumption?
- Does it reinforce existing biases?
- Is it based on a broad generalisation?
- Does it perpetuate divisions?

Step 4: Final Classification
Based on the analysis above, classify as:
- REGIONAL BIAS (1) if the comment reinforces regional biases or stereotypes
- NON-REGIONAL BIAS (0) if the comment is neutral, factual, or doesn't contain regional bias.

Your response must include a brief line of reasoning followed by the final classification in the format "Classification: [0 or 1]".
"""

set_seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

def load_data(csv_path):
    # Loads Dataset
    print(f"Loading dataset from {csv_path}.")
    try:
        df = pd.read_csv(csv_path)
        if 'level-1' not in df.columns or 'comment' not in df.columns:
             raise ValueError("Dataset must contain 'level-1' and 'comment' columns.")
             
        df['level-1'] = df['level-1'].astype(int)
        df['comment'] = df['comment'].astype(str)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

def format_prompt(row, tokenizer):
    # Formats a row into the Qwen chat template.
    label_text = CONFIG["label_map"][row['level-1']]
    
    messages = [
        {"role": "user", "content": f"{MODEL_PROMPT}\n\nComment: \"{row['comment']}\""},
        {"role": "assistant", "content": label_text}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def train_fold(fold_idx, train_df, val_df, tokenizer):
    # Trains a model for a specific fold.
    fold_output_dir = os.path.join(CONFIG["output_root_dir"], f"fold_{fold_idx}")
    print(f"\n--- Training Fold {fold_idx} ---")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_data = train_dataset.map(lambda x: format_prompt(x, tokenizer), num_proc=4)
    val_data = val_dataset.map(lambda x: format_prompt(x, tokenizer), num_proc=4)

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=os.path.join(fold_output_dir, "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="adamw_8bit",
        logging_steps=25,
        learning_rate=2e-5, 
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
    )

    trainer.train()
    
    adapter_path = os.path.join(fold_output_dir, "best_adapter")
    trainer.save_model(adapter_path)
    metrics = trainer.evaluate()
    val_loss = metrics['eval_loss']
    
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return adapter_path, val_loss

def parse_model_response(response_text):
    # Parses Classification: 0 or 1 from response.
    match = re.search(r'Classification:\s*([01])', response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if "regional bias" in response_text.lower() and "non-regional bias" not in response_text.lower():
        return 1
    if "non-regional bias" in response_text.lower():
        return 0
    if re.search(r'\b1\b', response_text):
        return 1
    if re.search(r'\b0\b', response_text):
        return 0
        
    return 0 # Default

def evaluate_fold(fold_idx, adapter_path, val_df, tokenizer):
    # Evaluates the trained fold model.
    print(f"\n--- Evaluating Fold {fold_idx} ---")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    predictions = []
    true_labels_int = val_df['level-1'].tolist()
    
    dataset = Dataset.from_pandas(val_df)

    for example in tqdm(dataset, desc=f"Inferencing Fold {fold_idx}"):
        messages = [
            {"role": "user", "content": f"{MODEL_PROMPT}\n\nComment: \"{example['comment']}\""}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        pred_int = parse_model_response(response)
        predictions.append(pred_int)

    # Metrics
    report = classification_report(true_labels_int, predictions, target_names=["Non-Regional Bias (0)", "Regional Bias (1)"], output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    acc = accuracy_score(true_labels_int, predictions)

    # Save Predictions
    res_df = val_df.copy()
    res_df['predicted_label'] = predictions
    res_df.to_csv(os.path.join(CONFIG["output_root_dir"], f"fold_{fold_idx}", "predictions.csv"), index=False)
    
    # Save Report
    report_text = classification_report(true_labels_int, predictions, target_names=["Non-Regional Bias (0)", "Regional Bias (1)"])
    with open(os.path.join(CONFIG["output_root_dir"], f"fold_{fold_idx}", "classification_report.txt"), "w") as f:
        f.write(report_text)
        
    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

    return f1_macro, acc

def main():
    if not os.path.exists(CONFIG["output_root_dir"]):
        os.makedirs(CONFIG["output_root_dir"])
        
    df = load_data(CONFIG["dataset_path"])
    tokenizer = get_tokenizer(CONFIG["model_id"])

    skf = StratifiedKFold(n_splits=CONFIG["num_folds"], shuffle=True, random_state=CONFIG["seed"])
    
    fold_results = []
    best_f1 = 0.0
    best_fold_idx = -1

    for fold_idx, (train_index, val_index) in enumerate(skf.split(df, df['level-1'])):
        current_fold = fold_idx + 1
        
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        fold_dir = os.path.join(CONFIG["output_root_dir"], f"fold_{current_fold}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        train_df.to_csv(os.path.join(fold_dir, "train_indices.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val_indices.csv"), index=False)

        # Train & Evaluate
        adapter_path, val_loss = train_fold(current_fold, train_df, val_df, tokenizer)
        f1, acc = evaluate_fold(current_fold, adapter_path, val_df, tokenizer)
        
        print(f"Fold {current_fold} Results -> F1 (Macro): {f1:.4f}, Accuracy: {acc:.4f}")
        
        fold_results.append({
            "fold": current_fold,
            "f1_macro": f1,
            "accuracy": acc,
            "val_loss": val_loss
        })

        if f1 > best_f1:
            best_f1 = f1
            best_fold_idx = current_fold
            best_model_path = os.path.join(CONFIG["output_root_dir"], "best_model_overall")
            if os.path.exists(best_model_path):
                shutil.rmtree(best_model_path)
            shutil.copytree(adapter_path, best_model_path)
            print(f"*** New Best Model found (Fold {current_fold})! Saved to {best_model_path} ***")

    # Summary
    results_df = pd.DataFrame(fold_results)
    results_df.loc['Average'] = results_df.mean()
    results_df.to_csv(os.path.join(CONFIG["output_root_dir"], "cv_metrics_summary.csv"))
    
    print("\n=============================================")
    print("   5-FOLD CV COMPLETE (Mistral 7B)")
    print("=============================================")
    print(results_df)

if __name__ == "__main__":
    main()
