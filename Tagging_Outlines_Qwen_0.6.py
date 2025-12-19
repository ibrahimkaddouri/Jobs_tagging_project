
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import outlines
import time
import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import subprocess
import numpy as np
import json
import unicodedata
import pickle as pkl
from typing import Literal, List
from tqdm import tqdm
from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# ## Cleaning data

def xstr(obj):
    return obj or ""

def clean_obj(obj):
    # remove unicode issues
    x = json.dumps(obj)
    x = re.sub(r"\\\\(?=u[0-9a-f]{4})", r"\\", x)
    x = json.loads(x)
    return x

def clean_text(text, max_linejumps = 1):
    # remove "\t" and "\r"
    cleaned_text = re.sub("\t", "", text)
    cleaned_text = re.sub("\r", "", cleaned_text)

    # remove duplicated spaces
    cleaned_text = re.sub(" +", " ", cleaned_text)

    # remove spaces around linejumps
    cleaned_text = re.sub(" +(?=\n)|(?<=\n) +", "", cleaned_text)

    # remove starting or ending linejumps and spaces
    cleaned_text = re.sub("^\n+|\n+$", "", cleaned_text)

    # remove duplicated linejumps
    cleaned_text = re.sub(
        "\n{" + str(max_linejumps) + ",}", "\n" * max_linejumps, cleaned_text
    )
    return cleaned_text

def normalize(item):
    return unicodedata.normalize("NFKD", item)

def transform_job(obj):
    obj = clean_obj(obj)
    n_sections = 6

    # Format
    job = {
        'title': xstr(obj.get('name', '')),
        'summary': xstr(obj.get('summary', '')),
        'sections': [
            {
                'title': section.get('title'),
                'description': section.get('description')
            }
            for section in obj.get('sections') or [] 
            if section.get('title') or section.get('description')
        ]
    }
    for title in ["requirements", "responsibilities", "culture", "benefits", "interviews"]:
        value = obj.get(title)
        if value is not None:
            job["sections"].append({"title": title.upper(), "description": value})

    # Structure
    title = job["title"]
    summary = job["summary"]
    sections = []
    for sec in job["sections"][:n_sections]:
        sec_text = ""
        # title
        if sec["title"]:
            sec_text += f"TITLE: {sec['title']}\n"
        # description
        if sec["description"]:
            sec_text += f"DESCRIPTION: {sec['description']}\n"
        sections.append(sec_text)

    # Merge
    output_body = ""
    if clean_text(title):
        output_body += f"TITLE\n{clean_text(title)}\n\n"
    if clean_text(summary):
        output_body += f"SUMMARY\n{clean_text(summary)}\n\n"

    for index, sec in enumerate(sections):
        if sec:
            output_body += f"SECTION {index}\n{clean_text(sec)}\n\n"

    output_body = normalize(output_body).strip()
    return output_body


# ## Reading jobs

# 1. Load the data
with open('/home/ibrahim/Downloads/tagging_dataset/job_key_to_json.pkl', 'rb') as f:
    raw_jobs = pkl.load(f)

cleaned_jobs_dict = {}

for job_id, job_obj in raw_jobs.items():
    try:
        # Apply cleaning pipeline
        cleaned_text = transform_job(job_obj)

        # Store in our new variable
        cleaned_jobs_dict[job_id] = cleaned_text
    except Exception as e:
        print(f"Error processing job {job_id}: {e}")

# ## Reading Tags

base_path = "/home/ibrahim/Downloads/tagging_dataset/open_ai_generations"

tags_dict = {}

for key in cleaned_jobs_dict.keys():
    filename = f"{key}.json"
    file_path = os.path.join(base_path, filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)

            tags_dict[key] = file_content

    except FileNotFoundError:
        print(f"Error processing file: {filename} (File not found)")

# ## Create the four tags datasets

education_gold_dict = {}
experience_gold_dict = {}
contract_gold_dict = {}
language_gold_dict = {}

for file_key, content in tags_dict.items():
    education_gold_dict[file_key] = content.get('Education')
    experience_gold_dict[file_key] = content.get('Experience')
    contract_gold_dict[file_key] = content.get('Contract')
    language_gold_dict[file_key] = content.get('Language')

# ## Cleaning the labels

to_replace = ['other', 'Franchise', 'Contract']

for key, value in contract_gold_dict.items():
    if value in to_replace:
        contract_gold_dict[key] = 'Other'

unique_values = set(contract_gold_dict.values())

experience_mapping = {
    'other': 'Other',
    'Associate': 'Associate (1–3 years)'
}

for key, value in experience_gold_dict.items():
    if value in experience_mapping:
        experience_gold_dict[key] = experience_mapping[value]

updated_values = set(experience_gold_dict.values())

# ## Data for the current experience

n = 3000 # Number of samples to use
keys_slice = list(islice(language_gold_dict, n))

# Create the subsets: One set of jobs, and four sets of tags
Language_truth = {k: language_gold_dict[k] for k in keys_slice}
Experience_truth = {k: experience_gold_dict[k] for k in keys_slice}
Education_truth = {k: education_gold_dict[k] for k in keys_slice}
Contract_truth = {k: contract_gold_dict[k] for k in keys_slice}
jobs = {k: cleaned_jobs_dict[k] for k in keys_slice}


# ## Loading the model

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 
)

model_1 = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_config 
)

model = outlines.from_transformers(model_1, tokenizer)


# ## Prediction of the education level

labels_educ = list(set(Education_truth.values()))

generator_educ_tag = outlines.Generator(model, eval('Literal'+str(labels_educ)))

tags_education = {}

for job_id, job_text in tqdm(jobs.items(), desc="Tagging Education"):
    try:
        prompt = f"""Task: Identify the required education level for the job description below.

Job Description:
{job_text}
"""
        result = generator_educ_tag(prompt)

        tags_education[job_id] = result

    except Exception as e:
        print(f"\nError on job {job_id}: {e}")
        tags_education[job_id] = "Other"

# ## Model evaluation

keys = Education_truth.keys()

y_true = [Education_truth[k] for k in keys]
y_pred = [tags_education[k] for k in keys]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# --- 1. Sauvegarde des métriques dans un fichier texte ---
with open("classification_results_Educ.txt", "w") as f:
    f.write("=== CLASSIFICATION METRICS ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1 Score: {f1:.4f}\n")
    
    f.write("\n=== CLASSIFICATION REPORT ===\n")
    f.write(classification_report(y_true, y_pred))

print("Les métriques ont été sauvegardées dans 'classification_results_Educ.txt'")

# --- 2. Sauvegarde de la Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Education Level Classification Confusion Matrix')

# Sauvegarde de l'image avant plt.show()
plt.savefig("confusion_matrix_Educ.png", dpi=300, bbox_inches='tight')

print("La heatmap a été sauvegardée sous 'confusion_matrix_Educ.png'")

# ## Using the model for the experience level

labels_exp = list(set(Experience_truth.values()))


generator_exp_tag = outlines.Generator(model, eval('Literal'+str(labels_exp)))

tags_experience = {}

for job_id, job_text in tqdm(jobs.items(), desc="Tagging Experience"):
    try:
        prompt = f"""Task: Identify the required experieEducation_truthnce level for the job description below.

Job Description:
{job_text}
"""
        result = generator_exp_tag(prompt)

        tags_experience[job_id] = result

    except Exception as e:
        print(f"\nError on job {job_id}: {e}")
        tags_experience[job_id] = "Other" 

print(f"Finished! Tagged {len(tags_experience)} jobs.")

keys = Experience_truth.keys()

y_true = [Experience_truth[k] for k in keys]
y_pred = [tags_experience[k] for k in keys]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# --- 1. Sauvegarde des métriques dans un fichier texte ---
with open("classification_results_Exp.txt", "w") as f:
    f.write("=== CLASSIFICATION METRICS ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1 Score: {f1:.4f}\n")
    
    f.write("\n=== CLASSIFICATION REPORT ===\n")
    f.write(classification_report(y_true, y_pred))

print("Les métriques ont été sauvegardées dans 'classification_results_Exp.txt'")

# --- 2. Sauvegarde de la Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Experience Level Classification Confusion Matrix')

# Sauvegarde de l'image avant plt.show()
plt.savefig("confusion_matrix_Exp.png", dpi=300, bbox_inches='tight')

print("La heatmap a été sauvegardée sous 'confusion_matrix_Exp.png'")

# ## Using the model for the contract

labels_contr = list(set(Contract_truth.values()))

print("Number of labels:", len(labels_contr))
print(labels_contr)


generator_contr_tag = outlines.Generator(model, eval('Literal'+str(labels_contr)))

tags_contract = {}

for job_id, job_text in tqdm(jobs.items(), desc="Tagging Contract"):
    try:
        prompt = f"""Task: Identify the contract type for the job description below.

Job Description:
{job_text}
"""
        result = generator_contr_tag(prompt)

        tags_contract[job_id] = result

    except Exception as e:
        print(f"\nError on job {job_id}: {e}")
        tags_contract[job_id] = "Other"

print(f"Finished! Tagged {len(tags_contract)} jobs.")

keys = Contract_truth.keys()

y_true = [Contract_truth[k] for k in keys]
y_pred = [tags_contract[k] for k in keys]

cm = confusion_matrix(y_true, y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# --- 1. Sauvegarde des métriques dans un fichier texte ---
with open("classification_results_Contract.txt", "w") as f:
    f.write("=== CLASSIFICATION METRICS ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1 Score: {f1:.4f}\n")
    
    f.write("\n=== CLASSIFICATION REPORT ===\n")
    f.write(classification_report(y_true, y_pred))

print("Les métriques ont été sauvegardées dans 'classification_results_Contract.txt'")

# --- 2. Sauvegarde de la Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Contract type Classification Confusion Matrix')

# Sauvegarde de l'image avant plt.show()
plt.savefig("confusion_matrix_Contract.png", dpi=300, bbox_inches='tight')

print("La heatmap a été sauvegardée sous 'confusion_matrix_Contract.png'")

# ## Using the model for the language

labels_lang = list(set(Language_truth.values()))

print("Number of labels:", len(labels_lang))
print(labels_lang)

generator_lang_tag = outlines.Generator(model, eval('Literal'+str(labels_lang)))

tags_language = {}

for job_id, job_text in tqdm(jobs.items(), desc="Tagging Language"):
    try:
        prompt = f"""Task: Identify the language for the job description below.

Job Description:
{job_text}
"""
        result = generator_lang_tag(prompt)

        tags_language[job_id] = result

    except Exception as e:
        print(f"\nError on job {job_id}: {e}")
        tags_language[job_id] = "Other" 

print(f"Finished! Tagged {len(tags_language)} jobs.")


# ## Evaluation of the models

keys = Language_truth.keys()

y_true = [Language_truth[k] for k in keys]
y_pred = [tags_language[k] for k in keys]

cm = confusion_matrix(y_true, y_pred)

# 1. Basic Metrics
# For multi-class, we use 'macro' or 'weighted' averaging
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')


# --- 1. Sauvegarde des métriques dans un fichier texte ---
with open("classification_results_Language.txt", "w") as f:
    f.write("=== CLASSIFICATION METRICS ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1 Score: {f1:.4f}\n")
    
    f.write("\n=== CLASSIFICATION REPORT ===\n")
    f.write(classification_report(y_true, y_pred))

print("Les métriques ont été sauvegardées dans 'classification_results_Language.txt'")

# --- 2. Sauvegarde de la Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Language Classification Confusion Matrix')

# Sauvegarde de l'image avant plt.show()
plt.savefig("confusion_matrix_Language.png", dpi=300, bbox_inches='tight')

print("La heatmap a été sauvegardée sous 'confusion_matrix_Language.png'")

def main():
    # Place the main logic of your code here
    print("Running my classification script...")

if __name__ == "__main__":
    main()