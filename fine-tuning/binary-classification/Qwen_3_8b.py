import os
import torch
import pandas as pd
import numpy as np
import gc
import shutil
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-8B" 

DATASET_PATH = "" # Dataset file path
OUTPUT_ROOT_DIR = "" # Output Directory
TARGET_GPU = "" # Target GPU node 
SEED = 42
MAX_SEQ_LENGTH = 2048 
MAX_NEW_TOKENS = 150 
NUM_CHUNKS = 10 
NUM_FOLDS = 5

def format_example(row, is_test=False):
    prompt = f"Analyze the following comment and classify it as Regional Bias (1) or Non-Regional Bias (0).\n\nComment: {row['comment']}\n\nClassification:"
    
    if is_test:
        return {"prompt": prompt, "label": row['is RB?']}
    else:
        return {"text": f"{prompt} {row['is RB?']}"}

def setup_env():
    # Sets up GPU visibility.
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)

def load_data():
    # Load the Dataset
    print(f"Loading dataset from {DATASET_PATH}.")
    df = pd.read_csv(DATASET_PATH)
    if 'is RB?' not in df.columns or 'comment' not in df.columns:
        raise ValueError("Dataset missing 'is RB?' or 'comment' columns.")
    df['is RB?'] = df['is RB?'].astype(int)
    df['comment'] = df['comment'].astype(str).str.strip()
    df = df.dropna(subset=['comment'])
    df = df.reset_index(drop=True) 
    return df

def get_model_and_tokenizer():
    # Loads the full model.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 
        
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model {MODEL_ID}: {e}")

def predict_and_save(model, tokenizer, df, fold_dir, prefix):
    # Predictions and Saving the model.
    print(f"Running inference on {prefix.upper()} set for {fold_dir}.")
    
    predictions = []
    batch_size = 8 
    prompts = [format_example(row, is_test=True)["prompt"] for _, row in df.iterrows()]
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating {prefix}"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for text in decoded:
            if "1" in text:
                predictions.append(1)
            elif "0" in text:
                predictions.append(0)
            elif "regional bias" in text.lower() and "non" not in text.lower():
                predictions.append(1)
            else:
                predictions.append(0) 
    
    # Save Metrics & Files
    y_true = df['is RB?'].tolist()
    
    # Save Predictions CSV
    res_df = df.copy()
    res_df['predicted_label'] = predictions
    csv_path = os.path.join(fold_dir, f"{prefix}_predictions.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"Saved {prefix} predictions to {csv_path}")
    
    # Metrics
    acc = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='macro', zero_division=0)
    
    metrics = {
        f"{prefix}_Accuracy": acc,
        f"{prefix}_F1_Macro": f1,
        f"{prefix}_Precision_Macro": precision,
        f"{prefix}_Recall_Macro": recall
    }
    
    # Save Classification Report
    report = classification_report(y_true, predictions, target_names=["Non-Regional Bias (0)", "Regional Bias (1)"])
    txt_path = os.path.join(fold_dir, f"{prefix}_classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"Saved {prefix} report to {txt_path}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Bias (0)', 'Bias (1)'], 
                yticklabels=['Non-Bias (0)', 'Bias (1)'])
    plt.title(f'{prefix.capitalize()} Confusion Matrix - Fold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    png_path = os.path.join(fold_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Saved {prefix} confusion matrix to {png_path}")
    
    return metrics

def train_and_eval_fold(fold_idx, train_df, val_df, test_df):
    fold_dir = os.path.join(OUTPUT_ROOT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    
    print(f"\n=== Processing Fold {fold_idx} ===")
    
    # Save Train/Val/Test Datasets
    train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
    
    # Prepare Data
    train_ds = Dataset.from_pandas(train_df).map(lambda x: format_example(x, is_test=False))
    val_ds = Dataset.from_pandas(val_df).map(lambda x: format_example(x, is_test=False))
    
    model, tokenizer = get_model_and_tokenizer()
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    training_args = SFTConfig(
        output_dir=os.path.join(fold_dir, "checkpoints"),
        num_train_epochs=10, 
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=8, 
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=20,
        optim="adamw_8bit", 
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        max_steps=-1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True 
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds, 
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)] 
    )
    
    print("Starting Training.")
    trainer.train()
    
    # Save Best Adapter
    adapter_path = os.path.join(fold_dir, "best_adapter")
    trainer.save_model(adapter_path)
    
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Evaluation Phase
    print("Loading best model for evaluation.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Evaluate Validation Set
    val_metrics = predict_and_save(model, tokenizer, val_df, fold_dir, "val")
    print(f"Fold {fold_idx} Val Metrics: {val_metrics}")
    
    # Evaluate Test Set
    test_metrics = predict_and_save(model, tokenizer, test_df, fold_dir, "test")
    print(f"Fold {fold_idx} Test Metrics: {test_metrics}")
    
    # Merge metrics
    combined_metrics = {**val_metrics, **test_metrics}
    combined_metrics["Fold"] = fold_idx
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return combined_metrics

def main():
    setup_env()
    df = load_data()
    
    print(f"\nSplitting dataset into {NUM_CHUNKS} stratified chunks for rotation.")
    skf = StratifiedKFold(n_splits=NUM_CHUNKS, shuffle=True, random_state=SEED)

    chunks_indices = []
    for _, chunk_idx in skf.split(df.index, df['is RB?']):
        chunks_indices.append(chunk_idx)
    
    fold_results = []
    
    for fold_idx in range(NUM_FOLDS): 
        current_fold = fold_idx + 1
        test_chunk_ids = [2*fold_idx, 2*fold_idx + 1]
        val_chunk_id = (2*fold_idx + 2) % NUM_CHUNKS
        
        train_chunk_ids = []
        for i in range(NUM_CHUNKS):
            if i not in test_chunk_ids and i != val_chunk_id:
                train_chunk_ids.append(i)
                
        print(f"\nConfiguration for Fold {current_fold}:")
        print(f"  Test Chunks: {test_chunk_ids}")
        print(f"  Val Chunk:   {[val_chunk_id]}")
        print(f"  Train Chunks: {train_chunk_ids}")
        
        test_indices = np.concatenate([chunks_indices[i] for i in test_chunk_ids])
        val_indices = chunks_indices[val_chunk_id]
        train_indices = np.concatenate([chunks_indices[i] for i in train_chunk_ids])
        
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        metrics = train_and_eval_fold(current_fold, train_df, val_df, test_df)
        fold_results.append(metrics)

    results_df = pd.DataFrame(fold_results)

    cols = ["Fold"] + [c for c in results_df.columns if c.startswith("test")] + [c for c in results_df.columns if c.startswith("val")]
    results_df = results_df[cols]
    
    avg_row = results_df.mean(numeric_only=True)
    avg_row["Fold"] = "Average"
    final_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    print("\n=== 5-Fold Cross-Validation Results ===")
    print(final_df.to_string(index=False))
    
    # Save Summary
    final_df.to_csv(os.path.join(OUTPUT_ROOT_DIR, "cv_metrics_summary.csv"), index=False)
    
if __name__ == "__main__":
    main()
