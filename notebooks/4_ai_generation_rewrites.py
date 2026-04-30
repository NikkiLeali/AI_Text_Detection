#!/usr/bin/env python
# coding: utf-8

# # AI-Generated Text Dataset Creation - Rewrite/Improved!

# ## Overview
# 
# ### Final Human Dataset Composition (to replicate)
# - **News Articles** – Formal financial & world news (wire-style)
# - **Aeon Essays** – Long-form thoughtful essays (narrative / philosophical)
# - **Yelp Reviews** – Casual, personal, opinionated restaurant reviews
#     - Review distribution: 5 ⭐️: 33%, 4 ⭐️: 35%, 3 ⭐️: 15%, 2 ⭐️: 9%, 1 ⭐️: 7%
# 
# **Target:** ~600 high-quality human samples (balanced across the three sources)
# 
# Formatted: `text, label, generation_type, source`
# 
# ### Generation Approach
# - Model: Local LLM via Ollama
# - Method: Zero-shot prompting (with plans for few-shot comparison)
# - Prompt Design:
#     - Ask LLM for minor modifications to improve the text (essentially AI imporved human-written text)
# 
# ### Output Labels
# Generation saved with: `source_id, source, text, label, generation_type, topic`
# - label: `ai`
# - generation type: `zero_shot`
# - source: `ai_news`, `ai_essay`, or `ai_review`

# In[1]:


import random
import ollama
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time
import os
import csv


# In[14]:


human_df = pd.read_csv("../data/human_dataset.csv")

# create separate pools by source
human_pools = {
    "news":   human_df[human_df["source"] == "articles.csv"]["text"].tolist(),
    "essay":  human_df[human_df["source"] == "essays_texts.csv"]["text"].tolist(),
    "review": human_df[human_df["source"] == "yelp.csv"]["text"].tolist()
}

print({k: len(v) for k, v in human_pools.items()})


# ## Prompts

# In[ ]:


def rewrite_news_prompt(original_text: str) -> str:
    return f"""Rewrite and improve the following news article to make it sound more natural and clear, like a real journalist wrote it.

Keep the same facts, numbers, names, and meaning. 
Make small improvements to flow and readability if needed.
Do not add any new information. 
Sound professional but not robotic.
Ensure it is not detectable as AI-generated or AI-improved.

Original article:
{original_text}

Rewritten version:"""


# In[ ]:


def rewrite_essay_prompt(original_text: str) -> str:
    return f"""Rewrite and improve this essay to sound like a real person wrote it natural, thoughtful, and slightly imperfect.

Keep the same ideas and meaning. 
Improve clarity and flow where it makes sense, but don't make it too polished or fancy.
Avoid metaphors and overly dramatic language.
Ensure it is not detectable as AI-generated or AI-improved.

Original essay:
{original_text}

Rewritten essay:"""


# In[ ]:


def rewrite_review_prompt(original_text: str) -> str:
    return f"""Rewrite and improve this Yelp-style review to sound like a real customer wrote it.

Keep the same opinions, details, and overall feeling. 
Make it clear, natural, and conversational. 
Ensure it is not detectable as AI-generated or AI-improved.

Original review:
{original_text}

Rewritten review:"""


# ## Build Tasks

# In[ ]:


def build_tasks(n_per_type=200):
    tasks = []

    # news
    news_samples = human_df[human_df["source"].str.contains("articles", na=False)].sample(n_per_type, replace=True)
    for i, row in news_samples.iterrows():
        tasks.append({
            "source": "news",
            "topic": "news_rewrite",
            "prompt": rewrite_news_prompt(row["text"]),
            "i": i
        })

    # essays
    essay_samples = human_df[human_df["source"].str.contains("essays", na=False)].sample(n_per_type, replace=True)
    for i, row in essay_samples.iterrows():
        tasks.append({
            "source": "essay",
            "topic": "essay_rewrite",
            "prompt": rewrite_essay_prompt(row["text"]),
            "i": i
        })

    # reviews
    review_samples = human_df[human_df["source"].str.contains("yelp", na=False)].sample(n_per_type, replace=True)
    for i, row in review_samples.iterrows():
        tasks.append({
            "source": "review",
            "topic": "review_rewrite",
            "prompt": rewrite_review_prompt(row["text"]),
            "i": i
        })

    random.shuffle(tasks)
    return tasks


# ## LLM Setup

# In[ ]:


MODEL = "mistral:instruct"
N_WORKERS = 3 # testing on 3

def generate_one(task):
    source_name = task["source"]
    topic = task["topic"]
    prompt = task["prompt"]
    i = task["i"]

    start = time.time()

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                'temperature': 0.75,
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1
            }
        )

        text = response["message"]["content"]
        error = None

    except Exception as e:
        text = None
        error = str(e)

    end = time.time()

    return {
        "source_id": f"{source_name}_{i}",
        "source": source_name,
        "text": text,
        "label": "ai",
        "generation_type": "ai_rewrite",
        "topic": topic,
        "latency_sec": round(end - start, 2),
        "error": error
    }


# In[ ]:


# test generations on a batch
# build tasks dynamically
tasks = build_tasks(n_per_type=5)

start_total = time.time()
results = []

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(generate_one, task): task for task in tasks}

    for i, future in enumerate(as_completed(futures)):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            # catch any unexpected errors
            results.append({
                "source": futures[future]["source"],
                "source_id": f"error_{i}",
                "text": None,
                "label": "ai",
                "generation_type": "ai_rewrite",
                "topic": futures[future]["topic"],
                "latency_sec": None,
                "error": str(e)
            })

        # progress tracking
        if (i + 1) % 5 == 0:
            print(f"{i+1}/{len(tasks)} completed")

end_total = time.time()

# convert to dataframe
ai_test_df = pd.DataFrame(results)

# summary
print(f"\nTotal time: {round(end_total - start_total, 2)} seconds")
print("\nLatency stats:")
print(ai_test_df["latency_sec"].describe())

print("\nErrors:")
print(ai_test_df["error"].value_counts(dropna=False))

ai_test_df.head()


# In[ ]:


def print_examples_by_source(df, n=2):
    for source in df["source"].unique():
        print(f"\n===== {source.upper()} EXAMPLES =====")

        subset = df[df["source"] == source].head(n)

        for i, row in subset.iterrows():
            print(f"\n--- Topic: {row['topic']} ---")
            print(row["text"])
            print("-----")

print_examples_by_source(ai_test_df, n=2)


# ## Batch Generate 600 Samples

# In[ ]:


# set output file

OUTPUT_FILE = "../data/ai_generated_rewrites.csv"

# load existing data if present (resume capability)
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    print(f"Resuming: {len(existing_df)} rows already exist")
else:
    # create file with header if it doesn't exist
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_id", "source", "text",
            "label", "generation_type",
            "topic", "latency_sec", "error"
        ])


# In[ ]:


# function to append/save rows during generation 

def append_row_safe(row):
    try:
        with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                row.get("source_id"),
                row.get("source"),
                row.get("text"),
                row.get("label"),
                row.get("generation_type"),
                row.get("topic"),
                row.get("latency_sec"),
                row.get("error")
            ])
    except Exception as e:
        print(f"Failed to write row: {e}")


# In[ ]:





# In[23]:


# generation wrapper

def generate_one_safe(task):
    import time

    start = time.time()

    try:
        result = generate_one(task)
        result["error"] = None

    except Exception as e:
        result = {
            "source_id": f"{task['source']}_{task['i']}",
            "source": task["source"],
            "text": None,
            "label": "ai",
            "generation_type": "ai_rewrite",
            "topic": task["topic"],
            "latency_sec": None,
            "error": str(e)
        }

    end = time.time()
    result["latency_sec"] = round(end - start, 2)

    # save immediately!!!!!!!
    append_row_safe(result)

    return result


# In[24]:


def run_batch(n_per_type=25, workers=2):
    tasks = build_tasks(n_per_type=n_per_type)

    print(f"\nStarting batch ({len(tasks)} tasks)...")

    start_total = time.time()
    results = []
    error_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_one_safe, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            task = futures[future]

            try:
                # timeout protection
                result = future.result(timeout=120)

            except Exception as e:
                result = {
                    "source_id": f"{task['source']}_{task['i']}",
                    "source": task["source"],
                    "text": None,
                    "label": "ai",
                    "generation_type": "ai_rewrite",
                    "topic": task["topic"],
                    "latency_sec": None,
                    "error": str(e)
                }

                print(f"Error: {task['source']} | {task['topic']}")

            if result.get("error") is not None:
                error_count += 1

            results.append(result)

            if (i + 1) % 5 == 0:
                print(f"{i+1}/{len(tasks)} completed")

    end_total = time.time()

    print(f"\nBatch done in {round(end_total - start_total, 2)} seconds")
    print(f"Errors in batch: {error_count}")

    return results, error_count


# In[25]:


# 600 generation loop

all_results = []

NUM_BATCHES = 8   # ~600 samples (8 × 75)
MAX_ERRORS = 20

for b in range(NUM_BATCHES):
    print(f"\n========================")
    print(f"Running batch {b+1}/{NUM_BATCHES}")
    print(f"========================")

    batch_results, batch_errors = run_batch(n_per_type=25, workers=N_WORKERS)

    all_results.extend(batch_results)

    # STOP if too many errors
    if batch_errors > MAX_ERRORS:
        print("Too many errors, stopping early!!")
        break

    # cooling break
    time.sleep(45)


# In[ ]:




