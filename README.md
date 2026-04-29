# Detecting AI-Generated Text Using Classical Machine Learning and LLM-Based Methods
**Team: Nikki Leali, Ella Protz, and Angelica Rodriguez**

## Overview

This project's goal is to 'develop and evaluate models that detect human-written text versus AI-generated text. The approach compares a classical machine learning pipeline using TF-IDF features and a Naïve Bayes classifier against a large language model (LLM)-based detection method. The project also examines whether AI-generated text created with few-shot prompting is more difficult to detect than zero-shot generated text.

## Objectives

* Build a labeled dataset of human-written and AI-generated text
* Train and evaluate a classical machine learning model using TF-IDF and Naïve Bayes
* Implement an LLM-based approach for text classification
* Compare performance between methods
* Analyze differences in detectability between zero-shot and few-shot generated text

## Setup
1. Create a virtual environment
2. pip install -r requirements.txt 
3. ...


## Dataset

The dataset will consist of at least 1,000 text samples:

* Human-written text collected from publicly available sources created prior to 2023
* AI-generated text created using large language models with both zero-shot and few-shot prompting

Each sample will include:

* The text content
* A label indicating human or AI-generated
* A generation type field (human, zero-shot, or few-shot)

## Methods

### 1. Data Collection and Preparation

* Collect human-written text from reliable public datasets
* Generate AI text using zero-shot prompts
* Generate AI text using few-shot prompts
* Combine all samples into a single structured dataset
* Clean and preprocess text (remove noise, normalize formatting if needed)
* Ensure balanced class distribution

### 2. Classical Machine Learning Approach

* Convert text into numerical features using TF-IDF
* Train a Naïve Bayes classifier
* Evaluate model performance using 10-fold cross-validation
* Record metrics including accuracy, precision, recall, and F1 score

### 3. LLM-Based Detection

* Design a prompt-based classification approach using an LLM
* Apply the model to classify text as human or AI-generated
* Store predictions and compare against true labels
* Compute the same evaluation metrics used for the classical model

### 4. Comparative Analysis

* Compare performance between the classical model and LLM-based method
* Evaluate differences in detecting zero-shot versus few-shot generated text
* Analyze patterns in misclassification
* Interpret results in terms of stylistic and statistical differences between text types

### 5. Results and Reporting

* Summarize findings in tables and visualizations
* Present key insights on model performance
* Discuss limitations and potential improvements
* Provide a clear conclusion addressing the project objectives

## Expected Outcome

The project will produce a comparative evaluation of classical and LLM-based methods for detecting AI-generated text, along with an analysis of how prompting strategies affect detectability.
