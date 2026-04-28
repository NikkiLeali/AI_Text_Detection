# Dataset

## Overview

The dataset contains human-written and AI-generated text used to evaluate text classification models.

## Format

Each row includes:

* `text`: the text sample
* `label`: `human` or `ai`
* `generation_type`: `human`, `zero_shot`, or `few_shot`

## Data Sources

* Human text: essays, online posts, and reviews from public sources
* AI text: generated using a language model with zero-shot and few-shot prompts

### Kaggle Human-Written Source Options
1. https://www.kaggle.com/code/erikbruin/nlp-on-student-writing-eda
2. https://www.kaggle.com/code/robikscube/student-writing-competition-twitch-stream 
3. https://www.kaggle.com/datasets/gpreda/ask-reddit
4. https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
5. https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles
6. https://www.kaggle.com/datasets/vivekhn/yelp-reviews

Likely essays, reddit, and reviews are the best choices for sourcing. 

## Notes

* Topics are consistent across human and AI text
* Text lengths are kept similar where possible
* Classes are balanced for fair evaluation
