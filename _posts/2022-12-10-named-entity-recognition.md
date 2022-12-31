---
title: "End-to-End Named Entity Recognition"
subtitle: "Build the First Web App by Transformers and Gradio"
date: 2022-12-10
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

## 2. PhoNER-COVID-19 Dataset
In this tutorial, I will use the [PhoNER-COVID-19](https://arxiv.org/abs/2104.03879v1) dataset, a dataset for recognizing COVID-19-related named entities in Vietnamese news, consisting of 35K entities over 10K sentences. The dataset includes 10 entity types with the aim of extracting key information related to COVID-19 patients, which are especially useful in downstream applications. In general, these entity types can be used in the context of not only the COVID-19 pandemic but also in other future epidemics: 

| Entity Type | Definition |
| :---------- | :--------- |
| PATIENT_ID          | Unique identifier of a COVID-19 patient in Vietnam. An PATIENT_ID annotation over "X" refers to as the X-th patient having COVID-19 in Vietnam. |
| NAME                | Name of a patient or person who comes into contact with a patient. |
| AGE                 | Age of a patient or person who comes into contact with a patient. |
| GENDER              | Gender of a patient or person who comes into contact with a patient. |
| JOB                 | Job of a patient or person who comes into contact with a patient. |
| LOCATION            | Locations/places that a patient was presented at. |
| ORGANIZATION        | Organizations related to a patient, e.g. company, government organization, and the like, with structures and their own functions. |
| SYMPTOM_AND_DISEASE | Symptoms that a patient experiences, and diseases that a patient had prior to COVID-19 or complications that usually appear in death reports. |
| TRANSPORTATION      | Means of transportation that a patient used. Here, we only tag the specific identifier of vehicles, e.g. flight numbers and bus/car plates. |
| DATE                | Any date that appears in the sentence. |

The dataset was randomly split into train/val/test sets with a ratio of 5/2/3, ensuring comparable distributions of entity types across these three sets. Statistics of the dataset are presented in the table below: 

| Entity Type | Train |   Val |  Test |   All |
| :---------- | ----: | ----: | ----: | ----: |
| PATIENT_ID            | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3240 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1276 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2005 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 6521 |
| NAME                  | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 349 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 188 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 318 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 855 |
| AGE                   | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 682 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 361 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 582 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1625 |
| GENDER                | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 542 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 277 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 462 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1281 |
| JOB                   | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 205 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 132 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 173 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 510 |
| LOCATION              | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5398 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2737 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4441 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 12576 |
| ORGANIZATION          | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1137 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 551 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 771 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2459 |
| SYMPTOM_AND_DISEASE   | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1439 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 766 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1136 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3341 |
| TRANSPORTATION        | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 226 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 87 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 193 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 506 |
| DATE                  | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2549 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1103 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1654 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5306 |
| # Entities in total   | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 15767 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 7478 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 11735 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 34984 |
| # Sentences in total  | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5027 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2000 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3000 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 10027 |

If you are building your own dataset, [Prodigy](https://prodi.gy/) is a great annotation tool for a NER task. 

## References
[[1] COVID-19 Named Entity Recognition for Vietnamese](https://arxiv.org/abs/2104.03879v1)<br>
{: style="font-size: 14px;"}