# Notebooks

## Structure

```
notebooks/
├── 0_human_data_prep.ipynb
├── 1_ai_generation.ipynb
├── 2_master_data_merge.ipynb
```

---

## Notebook Breakdown

### 0_human_data_prep.ipynb

Prepares and cleans human-written text data from multiple sources.

**Sources:**

* Aeon Essays (long-form writing)
* News Articles (formal, structured writing)
* Yelp Reviews (casual, opinionated writing)

**Key Steps:**

* Data loading and inspection
* Filtering by length and structure
* Removal of formatting artifacts
* Standardization of columns
* Final balanced dataset (200 per source)

**Output:**

```
data/human_dataset.csv
```

---

### 1_ai_generation.ipynb

Generates AI-written text that mirrors the human dataset in structure and style.

**Key Features:**

* Local LLM generation using Ollama
* Zero-shot prompting
* Dynamic prompt construction with:

  * Randomized topics
  * Mixed specificity (general + specific entities)
  * Variable constraints and tone
  * Style matching (news, essays, reviews)

**Design Goals:**

* Match human writing styles
* Avoid repetitive or overly polished outputs
* Maintain diversity across samples

**Generation Strategy:**

* 200 news-style samples
* 200 essay-style samples
* 200 review-style samples
* Batched generation with parallel execution
* Immediate CSV saving for fault tolerance

**Output:**

```
data/ai_generated_live.csv
```

---

### 2_master_data_merge.ipynb

Combines and finalizes the human and AI datasets into a single modeling-ready dataset.

**Key Steps:**

* Load human and AI datasets
* Remove invalid or empty rows
* Clean text (remove emojis / non-ASCII characters)
* Align columns across datasets
* Shuffle data for unbiased training
* Assign a unified `id` column

**Final Dataset Structure:**

* `id`
* `source`
* `text`
* `label` (human / ai)
* `generation_type`

**Output:**

```
data/final_dataset.csv
```
---

