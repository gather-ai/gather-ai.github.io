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
| PATIENT_ID            | 3240 | 1276 | 2005 | 6521 |
| NAME                  | 349 | 188 | 318 | 855 |
| AGE                   | 682 | 361 | 582 | 1625 |
| GENDER                | 542 | 277 | 462 | 1281 |
| JOB                   | 205 | 132 | 173 | 510 |
| LOCATION              | 5398 | 2737 | 4441 | 12576 |
| ORGANIZATION          | 1137 | 551 | 771 | 2459 |
| SYMPTOM_AND_DISEASE   | 1439 | 766 | 1136 | 3341 |
| TRANSPORTATION        | 226 | 87 | 193 | 506 |
| DATE                  | 2549 | 1103 | 1654 | 5306 |
| # Entities in total   | 15767 | 7478 | 11735 | 34984 |
| # Sentences in total  | 5027 | 2000 | 3000 | 10027 |

If you are annotating your own dataset, [Prodigy](https://prodi.gy/) is a great annotation tool for the NER task. 

## References
[[1] COVID-19 Named Entity Recognition for Vietnamese](https://arxiv.org/abs/2104.03879v1)<br>
{: style="font-size: 14px;"}