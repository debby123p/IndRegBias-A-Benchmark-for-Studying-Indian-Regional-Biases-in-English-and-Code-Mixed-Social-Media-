# CAN LARGE LANGUAGE MODELS DETECT INDIAN REGIONAL BIAS? DATASET & ANALYSIS
## Overview
This research examines the effectiveness of Large Language Models (LLMs) in detecting regional bias within social media comments in the Indian context. In a culturally diverse nation like India, regional biases can perpetuate stereotypes and create social divisions. This project addresses the challenge that many current LLMs are trained primarily on Western datasets, leading to a lack of awareness of the nuanced biases prevalent in India.
Our goal is to make LLMs aware of regional bias in Indian content and improve their detection performance. This work contributes a novel dataset, a methodology for bias detection, and provides insights into the strengths and limitations of current LLMs for this task.

## Dataset

A key contribution of this research is a new, culturally specific dataset of user-generated comments for detecting regional bias.

- Data Source: Comments collected from Reddit and YouTube.

- Final Size: After cleaning, the final dataset consists of 25,000 user comments.
  
- Annotation: A rigorous human annotation process was implemented, involving two groups of three annotators. Inter-annotator agreement was high, with a Cohen's kappa value of 0.91 for binary classification and 0.83 for multi-class classification.

### Annotation Schema
Each of the 25,000 comments was manually classified using a multi-level severity schema:


Level 1: Bias Identification: Is the comment a regional bias (1) or not (0). 


Level 2: Severity of Bias: The severity of the comment was rated as Mild (1), Moderate (2), or Severe (3).


Level 3: State Identification: The name of the Indian state being targeted in the comment was recorded.

### Data Analysis Highlights

These comments are collected from videos or subreddit pages belonging to different regions, where the languages are mixed, such as English, Hinglish (a mix of Hindi and English), a mix of Bengali and English, a mix of Malayalam and English, or Marathi and English; thus, we have a multilingual and code-mixed language dataset.

- Out of 25,000 comments,13,020 (52.1%) contained regional biases, and 11,980 (47.9%) contained non-regional biases.
  
- Severity breakdown of biased comments is as Mild (3,747 comments), Moderate (6,663 comments), and Severe (2,610 comments).

- The data also showed the distribution of comments across different regions of India, with a large number of biased comments found for South India (5,802 comments) and North India (3,906 comments).

- In the state-wise breakdown, Kerala emerged as the most represented state with 1,841 comments, followed by Goa (1,103) and West Bengal (1,072).

## Methodology

We deployed a series of experiments in the following setting with the Chain-of-thought (CoT) prompting technique for the classification of level-1 and level-2 annotations:

- Zero-Shot Setting

- Few-Shot Setting

- Fine-Tuning (Using Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA))
  
      - Instruction-Based Supervised Fine-Tuning (SFT) for Binary Classification
  
      - Classification-Based Supervised Fine-Tuning (SFT) for Multi-class Classification 

### Models Evaluated
Eight prominent LLMs were selected for their instruction-following and reasoning abilities:

- qwen/Qwen3-8B

- qwen/qwen3-32B
  
- mistralai/Mistral-7B-Instruct-v0.3

- mistralai/Mistral-Nemo-Base-2047

- meta-llama/Llama-3.2-3B 

- deepseek-ai/DeepSeek-R1-Distill-Llama-8B 

- google/gemma-1.1-7b-it

- google/gemini-2.5-pro

- microsoft/Phi-4-mini-reasoning(4b) 

## Results

### Results: Binary Classification 

1) Zero-Shot Results

   The results for the models are presented in the zero-shot setting in the table. The codes for which are available in the folder named zero-shot-binary-classification, which is in [zero-shot-binary-classification](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/tree/main/zero_shot/zero-shot-binary-classification).
   
   ![image](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/blob/main/Images/PresentationWork%20Progress.png)

3) Few-Shot Results

   We conducted the few-shot experiments with a smaller number of support (different combinations of regional biases and non-regional biases) on a smaller dataset of 1000 comments for the model Qwen3_8b. The codes for which are available in [qwen_3_8b](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/tree/main/few-shot).

   ![image](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/blob/main/Images/Binary%20Classification%20(13).png)

   After looking through the inferences of the above experiment, we proceeded with experiments on the entire dataset with the type of support we have provided in Exp-2, Exp-3, and Exp-4, as we got improved performances in comparison to the zero-shot results. The codes for which are , which is available in [Few-shot-binary-classification](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/tree/main/few-shot).

   ![image](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/blob/main/Images/PresentationWork%20Progress%20(1)%20(1).png)

   The few-shots results show improvement over the zero-shot results as present in the above table. The experiment with providing only 50 regional biases as support has outperformed other few-shot experiments. Also, balanced and unbalanced set of support show improvement in performance in comparison to the zero-shot.   

5) Fine-Tuning Results

   Fine-tuned Qwen3_32b has emerged as the absolute best performer, achieving high reliability with F1-scores and precision nearing 0.90 for both bias categories. Followed by Qwen3_8b model which has also performed better than zero-shot, achieving near 0.90 scores for precison and F1-scores. Fine-tuned Mistral showed only minor improvements over its zero-shot, demonstrating far less adaptability to this specific task compared to Qwen. The codes for which are available in [binary-classification](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/tree/main/fine-tuning).

   ![image](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/blob/main/Images/PresentationWork%20Progress%20(2)%20(1).png)

### Results: Multi-Class Classification

1) Zero-Shot Results

   ![image](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/blob/main/Images/Binary%20Classification%20(1).png)

3) Few-Shot Results

   As observed in the binary classification experiments, the few-shot performance was consistently poor in comparison to the zero-shot baseline. Consequently, we eliminated few-shot experiments for the multi-class            classification phase. This decision was driven by the significant imbalance in the dataset and the complexity required to curate varied support examples, which previous experiments showed led to model confusion rather     than improvement.

4) Fine-Tuning Results

   Needs to be updated.
   
## Conclusion

- Benchmark Creation: Developed a novel dataset to address the gap in regional bias detection resources.
  
- Prompting Analysis: Contrary to standard NLP trends, Few-Shot Learning proved detrimental, causing catastrophic forgetting and reduced generalisation compared to Zero-Shot settings.
  
- Fine-Tuning Efficacy: Utilised PEFT with LoRA, which significantly outperformed prompting strategies and effectively solved the binary bias detection task.
  
- Classification Challenges: While binary detection achieved high accuracy, Severity Classification remains a challenge due to the subtle nuances between severity levels.


## Future Work

- LLM vs. Human Annotation: Systematically compare LLM-generated annotations against our human-annotated gold standard to evaluate if LLMs can be trusted for this nuanced task. This is crucial for developing scalable methods for building large, culturally-aware datasets.

