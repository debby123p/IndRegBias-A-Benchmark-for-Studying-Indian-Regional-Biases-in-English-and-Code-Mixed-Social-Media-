import os
import torch
import pandas as pd
import re
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc

HF_API_KEY = "" # Hugging_face Token
TARGET_GPU = "" # Target GPU node
INPUT_CSV_PATH = "" # Dataset file path
OUTPUT_DIR = "" # Output Directory
COMMENT_COLUMN_NAME = "comment"
GROUND_TRUTH_COLUMN_NAME = "is RB?"
BATCH_SIZE = 16 

MODEL_ID = "krutrim-ai-labs/Krutrim-2-instruct" 

SYSTEM_PROMPT = """
You are an expert in identifying regional biases in social media comments about Indian states and regions. Your task is to classify whether a comment contains regional biases or not.

Task: Classify the given comment as either "REGIONAL BIAS" (1) or "NON-REGIONAL BIAS" (0).

Definition:
- REGIONAL BIAS (1): Comments that reinforce or propagate biases about Indian states, regions, or their people based on linguistic, cultural, economic, political, or infrastructural aspects. The comments can reflect either positive or negative biases towards specific states or regions.
- NON-REGIONAL BIAS (0): Comments that are neutral or factual without generalisations, or unrelated to regional characteristics.

Step-by-Step Analysis Process:
Step 1: Identify Regional References
Think: Does this comment mention or refer to:
- Specific Indian states (e.g., Bihar, Kerala, Punjab, etc.)
- Regional groups (e.g., South Indians, North Indians, Biharis, etc.)
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

def setup_environment():
    # Sets up GPU visibility and creates the output directory.
    print(f"Restricting execution to GPU: {TARGET_GPU}")
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU.split(":")[-1] 
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

def load_model_and_tokenizer():
    # Handles authentication and loads the model.
    print("Logging into Hugging Face Hub.")
    login(token=HF_API_KEY)

    print(f"Loading model: {MODEL_ID} in full precision (bfloat16).")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=False 
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left' 
            
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model {MODEL_ID}. Error: {e}")

def parse_single_response(response_text):
    # Robustly parses the model response to extract the classification (0 or 1).
    prediction = -1
    reasoning = response_text.split("Classification:")[0].strip() or response_text
  
    match = re.search(r'Classification:\s*([01])', response_text)
    if match:
        prediction = int(match.group(1))
    else:
        if re.search(r'REGIONAL BIAS|1', response_text, re.IGNORECASE):
            prediction = 1
        elif re.search(r'NON-REGIONAL BIAS|0', response_text, re.IGNORECASE):
            prediction = 0
    
    if prediction == -1:
        print(f"Warning: Could not parse model output. Snippet: '{response_text[:50]}.'")
        prediction = 0
        
    return prediction, reasoning

def classify_batch(comments, model, tokenizer):
    # Generates classifications for a batch of comments.
    
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Comment: \"{comment}\"\n\nBased on your analysis, provide your reasoning and final classification."}
        ] for comment in comments
    ]

    prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    results = []
    outputs = None 

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        for text in response_texts:
            results.append(parse_single_response(text))

    except Exception as e:
        print(f"An error occurred during model generation for a batch: {e}")
        results = [(0, f"Error: {e}")] * len(comments)

    if 'inputs' in locals(): del inputs
    if 'outputs' in locals() and outputs is not None: del outputs
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def generate_evaluation_outputs(df):
    # Generates classification report and confusion matrix.
    y_true = df[GROUND_TRUTH_COLUMN_NAME].astype(int)
    y_pred = df['predicted_label'].astype(int)

    report = classification_report(y_true, y_pred, target_names=["NON-REGIONAL BIAS (0)", "REGIONAL BIAS (1)"], zero_division=0)
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report for model: {MODEL_ID}\n\n")
        f.write(report)
    print(f"\nClassification report saved to {report_path}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["NON-REGIONAL BIAS (0)", "REGIONAL BIAS (1)"], 
                yticklabels=["NON-REGIONAL BIAS (0)", "REGIONAL BIAS (1)"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {MODEL_ID}')
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

def main():
    # Main function to orchestrate the entire classification process.
    setup_environment()
    
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return

    print(f"\nReading input CSV from: {INPUT_CSV_PATH}")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Dataset file '{INPUT_CSV_PATH}' not found.")
        return

    df = pd.read_csv(INPUT_CSV_PATH)

    if COMMENT_COLUMN_NAME not in df.columns:
        raise ValueError(f"Column '{COMMENT_COLUMN_NAME}' not found in dataset. Available columns: {df.columns.tolist()}")

    df[COMMENT_COLUMN_NAME] = df[COMMENT_COLUMN_NAME].astype(str).fillna("")
    comments_to_process = df[COMMENT_COLUMN_NAME].tolist()
    
    all_results = []
    
    print(f"Starting classification for {len(comments_to_process)} comments with batch size {BATCH_SIZE}.")
    
    for i in tqdm(range(0, len(comments_to_process), BATCH_SIZE), desc="Classifying batches"):
        batch_comments = comments_to_process[i:i + BATCH_SIZE]
        batch_results = classify_batch(batch_comments, model, tokenizer)
        all_results.extend(batch_results)
    
    df['predicted_label'] = [res[0] for res in all_results]
    df['model_reasoning'] = [res[1] for res in all_results]

    output_csv_path = os.path.join(OUTPUT_DIR, "classification_results_krutrim.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"\nClassification complete. Results saved to {output_csv_path}")
    
    if GROUND_TRUTH_COLUMN_NAME in df.columns:
        generate_evaluation_outputs(df)
    else:
        print(f"\nGround truth column '{GROUND_TRUTH_COLUMN_NAME}' not found. Skipping evaluation.")

if __name__ == "__main__":
    main()
