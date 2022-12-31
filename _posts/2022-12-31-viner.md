---
title: "End-to-End Named Entity Recognition"
subtitle: "Build the First Web App by Transformers and Gradio"
date: 2022-12-31
categories: 
  - Tutorials
tags: 
  - AI Applications
  - Natural Language Processing
header: 
  image: "/assets/images/viner/ner.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page. In the last half-decade, Natural Language Processing (NLP) applications appear more and more in industrial products or business processes, reaching the same popularity as Computer Vision. Therefore, this field is too big to ignore. In this post, we will first time talk about an important NLP problem, [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) (NER) which is the task of tagging entities in text with their corresponding type. In addition, we will also build a simple AI-core web application using [Gradio](https://gradio.app/) and Hugging Face [Spaces](https://huggingface.co/spaces). 

## 1. About NER and NLP

Where is NER in the big picture of NLP? NLP is a large field with lots of various tasks. However, every NLP task can be categorized into 4 main groups: 
* **Text Classification**: Similar to Image Classification, Text Classification is a task of assigning a set of predefined classes to a sequence (a sentence, paragraph, or whole document). Some of the most well-known examples of Text Classification include Sentiment Analysis, Topic Labeling, Language Detection, and Intent Detection. 
* **Text Tagging**: Text Tagging or Text Labeling is a core Information Extraction task in which each unique word (token) in a sequence is classified using a pre-defined label set. Text Tagging has some exciting applications such as Named Entity Recognition, or Part-of-Speech Tagging. 
* **A mix of Text Classification and Tagging**: Multi-task Learning was introduced many times on my page. This group of tasks is where a model is expected to classify a given sequence and tag every word of it simultaneously. Some examples of this group are Named Entity Recognition and Relation Extraction, Intent Detection and Slot Filling. 
* **Text Generation**: Text generation is the task of generating text with the goal of appearing indistinguishable from the human-written text. This task has many wonderful applications such as (Abstractive) Document Summarization, Machine Translation, or Chatbot. 

## 2. Dataset and Annotation Process