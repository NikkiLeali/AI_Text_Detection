# Dataset

## Overview

The dataset contains human-written and AI-generated text used to evaluate text classification models.

## Format

Each row includes:

* `source_id`: identification from source csv
* `source`: where the sample came from
* `text`: the text sample
* `label`: `human` or `ai`
* `generation_type`: `human`, `zero_shot`, or `few_shot`


## Data Sources

* Human text: essays, online posts, and reviews from public sources
* AI text: generated using a language model with zero-shot and few-shot prompts

### Kaggle Human-Written Source Options
1. Essays: https://www.kaggle.com/datasets/mannacharya/aeon-essays-dataset
2. Artcles: https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles
3. Yelp: https://www.kaggle.com/datasets/vivekhn/yelp-reviews

Human data split:
* 33% student essays
* 33% Yelp reviews
* 33% news articles

Final: 600 rows of cleaned text entries (200, 200, 200 split)

## Notes

* Topics are consistent across human and AI text
* Text lengths are kept similar where possible
* Classes are balanced for fair evaluation
