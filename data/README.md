# Dataset

## Overview

This dataset contains human-written and AI-generated text used to train and evaluate models for detecting AI-generated content. The dataset is designed to balance multiple writing styles (formal, semi-formal, and informal) to better reflect real-world variation in human text.

---

## Format

Each row includes:

* `source_id`: Unique identifier derived from the original dataset or generation process
* `source`: Origin of the sample (e.g., dataset name or AI generation type)
* `text`: The text sample
* `label`: `human` or `ai`
* `generation_type`: `human`, `zero_shot`, or `few_shot`

---

## Data Sources

### Human-Written Text

Human samples were collected from publicly available datasets:

1. **Essays (Aeon dataset)**
   https://www.kaggle.com/datasets/mannacharya/aeon-essays-dataset

2. **News Articles**
   https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles

3. **Yelp Reviews**
   https://www.kaggle.com/datasets/vivekhn/yelp-reviews

These sources provide a mix of:

* Formal writing (news articles)
* Semi-formal analytical writing (essays)
* Informal, conversational writing (reviews)


### AI-Generated Text

AI samples were generated locally using a large language model (via Ollama) with structured prompt templates.

Generation was controlled to match:

* Topic domains
* Writing styles (news, essay, review)
* Length distributions

Prompts included randomized:

* Topics and scenarios
* Writing styles and tones
* Structural constraints

This ensured diversity while maintaining consistency with human data.

---

## Dataset Composition

### Human Data

* 200 essay samples
* 200 news article samples
* 200 review samples

Total: **600 human samples**


### AI Data

* 200 essay-style samples
* 200 news-style samples
* 200 review-style samples

Total: **600 AI samples (zero-shot)**


### Final Dataset

* **Total samples:** 1,200
* Balanced across:

  * Human vs AI (600 / 600)
  * Writing styles (essay / news / review)

---

## Preprocessing

Human and AI text were cleaned to remove artifacts that could bias classification, including:

* Non-standard Unicode characters (e.g., emojis)
* Formatting artifacts from source datasets
* Corrupted or incomplete text entries

Text was otherwise left largely unaltered to preserve natural variation in writing style.

---

## Notes

* Topics are aligned between human and AI samples to prevent topic-based bias
* Text lengths are kept within similar ranges across categories
* The dataset intentionally includes variation in tone and structure to reflect realistic writing differences
* AI samples were generated using zero-shot prompting (few-shot generation may be added later for comparison)
