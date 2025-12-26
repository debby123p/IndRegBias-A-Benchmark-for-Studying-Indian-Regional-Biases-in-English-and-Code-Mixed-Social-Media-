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

HF_TOKEN = "" # Hugging_face Token
TARGET_GPU = "" # Target GPU node 
INPUT_CSV_PATH = "" # Dataset file path
OUTPUT_DIR = "" # Output Directory
COMMENT_COLUMN_NAME = "comment"
GROUND_TRUTH_COLUMN_NAME = "severity" 
BATCH_SIZE = 16 
MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407" # Model ID

MODEL_PROMPT = """
You are an expert in analysing the severity of regional biases in social media comments about Indian states and regions. You are provided with comments that have already been identified as containing regional bias. Your task is to determine the severity level of the bias present.

Task: Classify the severity of the regional bias in the comment as "SEVERE" (3), "MODERATE" (2), or "MILD" (1).

Definitions (Check in this order):
- LEVEL 3 (SEVERE): Comments that are overtly hostile, hateful, or derogatory. These include usage of regional slurs, dehumanising language, calls for exclusion (e.g., "Go back to your state"), or statements that promote hatred/violence against a specific region or group.
- LEVEL 2 (MODERATE): Comments that contain explicit negative generalisations, mockery, or clearly biased assumptions about a region's culture, language, or people. The tone is critical or mocking but does not incite violence or use extreme profanity/slurs.
- LEVEL 1 (MILD): Comments that contain subtle stereotypes, "benevolent" or positive biases (e.g., "People from State X are always smart"), or minor negative generalisations that are not aggressive. These comments rely on low-level regional tropes without expressing hostility.

Step-by-Step Analysis Process:
Step 1: Analyze the Stereotype or Generalization
Think: What specific regional claim is being made?
- Is it a positive generalisation?
- Is it a negative stereotype? 

Step 2: Assess Tone and Intent
Evaluate the emotional weight of the words:
- Is the tone aggressive, hateful, or threatening? (Check for Level 3 first)
- Is the tone mocking, sarcastic, or condescending? (Check for Level 2)
- Is the tone casual or "matter-of-fact"? (Check for Level 1)

Step 3: Check for Escalating Factors
Look for specific triggers:
- For Level 3: Does it contain slurs? Does it question citizenship/belonging? Is it dehumanising?
- For Level 2: Does it imply one group is superior to another?

Step 4: Final Classification
Based on the analysis above, assign the severity score:
- 3: If the bias is abusive, hateful, or extreme.
- 2: If the bias is explicit and negative, but not abusive.
- 1: If the bias is subtle, positive, or non-hostile.

Your response must include a brief line of reasoning followed by the final classification in the format "Classification: [1, 2, or 3]".
"""

def setup_environment():
    # Sets up GPU visibility and creates the output directory.
    print(f"Restricting execution to GPU: {TARGET_GPU}")
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

def load_model_and_tokenizer():
    # Handles authentication and loads the full model.

    print("Logging into Hugging Face Hub.")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    print(f"Loading model: {MODEL_ID} in bfloat16.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
  
    if tokenizer.chat_template is None:
        print("WARNING: No chat template detected. Assigning a default ChatML template.")
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
    return model, tokenizer

def parse_single_response(response_text):
    # Robustly parses a single model response text to ensure a 1, 2, or 3 output.
    prediction = -1
    reasoning = response_text.split("Classification:")[0].strip() or response_text
    match = re.search(r'Classification:\s*([123])', response_text)
    if match:
        prediction = int(match.group(1))
    else:
        digits = re.findall(r'\b([1-3])\b', response_text.split('\n')[-1])
        if digits:
            prediction = int(digits[-1])
    
    if prediction == -1:
        if response_text.strip().isdigit() and int(response_text.strip()) in [1, 2, 3]:
            prediction = int(response_text.strip())
        else:
            print(f"Warning: Could not parse model output. Defaulting to 1 (Mild). Response snippet: '{response_text[-50:]}'")
            prediction = 1
        
    return prediction, reasoning

def classify_batch(comments, model, tokenizer):
    # Generates classifications for a batch of comments.
    messages_batch = []
    for comment in comments:
        full_content = f"{MODEL_PROMPT}\n\nComment: \"{comment}\"\n\nBased on your analysis, provide your reasoning and final classification."
        messages_batch.append([{"role": "user", "content": full_content}])
    formatted_prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
    
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    
    results = []
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.1, 
                pad_token_id=tokenizer.eos_token_id
            )
        response_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        for text in response_texts:
            results.append(parse_single_response(text))

    except Exception as e:
        print(f"An error occurred during model generation for a batch: {e}")
        results = [(1, f"Error: {e}")] * len(comments)
    
    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def generate_evaluation_outputs(df):
    # Generates and saves the classification report and confusion matrix.
    y_true = df[GROUND_TRUTH_COLUMN_NAME].astype(int)
    y_pred = df['predicted_label'].astype(int)

    target_names = ["Mild Bias (1)", "Moderate Bias (2)", "Severe Bias (3)"]
    labels = [1, 2, 3]

    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report for model: {MODEL_ID}\n\n")
        f.write(report)
    print(f"\nClassification report saved to {report_path}")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {MODEL_ID}')
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

def main():
    # Main function to orchestrate the entire classification process.
    setup_environment()
    model, tokenizer = load_model_and_tokenizer()
    
    print(f"\nReading input CSV")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return

    if COMMENT_COLUMN_NAME not in df.columns:
        raise ValueError(f"Comment column error for '{COMMENT_COLUMN_NAME}'.")

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

    output_csv_path = os.path.join(OUTPUT_DIR, "classification_results.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"\nClassification complete. Results saved to {output_csv_path}")
    
    if GROUND_TRUTH_COLUMN_NAME in df.columns:
        generate_evaluation_outputs(df)
    else:
        print(f"\nGround truth column '{GROUND_TRUTH_COLUMN_NAME}' not found. Skipping evaluation.")

if __name__ == "__main__":
    main()
