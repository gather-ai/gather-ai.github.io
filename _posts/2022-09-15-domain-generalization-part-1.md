---
title: "Domain Generalization Tutorials (Part 1): Multi-task Learning and Flat Minima Seeking"
date: 2022-09-15
categories: 
  - Tutorials
tags: 
  - deep learning
header: 
  image: "/assets/images/rome.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. I'm Khiem. Welcome to my website, where I share some intuitive explanations and hands-on tutorials on a range of topics in AI. 
{: style="text-align: justify;"}

🚀 I am going to kick off this site with a series of tutorials about the topic of Domain Generalization. This series provides a systematic survey of outstanding methods in literature and my own implementations to demonstrate these methods. This is the first part of the series that gives you a brief understanding of the term Domain Generalization and introduces the first approaches to the problem. 
{: style="text-align: justify;"}

## 1. Background
Machine learning (ML) systems generally rely on an over-simplified assumption, that is, the training (source) and testing (target) data are independent and identically distributed (i.i.d.), however, this assumption is not always true in practice. When the distributions of training data and testing data are different, which is referred to as the domain shift problem, the performance of these ML systems often catastrophically decreases due to domain distribution gaps. Moreover, in many applications, target data is difficult to obtain or even unknown before deploying the model. For example, in biomedical applications where data differs from equipment to equipment and institute to institute, it is impractical to collect the data of all possible domains in advance. 
{: style="text-align: justify;"}

To address the domain shift problem, as well as the absence of target data, the topic of Domain Generalization (DG) was introduced. Specifically, the goal in DG is to learn a model using data from a single or multiple related but distinct source domains in such a way that the model can generalize well to any **_unseen_** target domain. 
{: style="text-align: justify;"}

**Watch out!** Unlike other related topics such as Domain Adaptation (DA) or Transfer Learning (TL), where the ML models can do some forms of adaptation on target data, DG considers the scenarios where target data is inaccessible during model learning. 
{: style="text-align: justify;"}
{: .notice}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/domain-generalization/DG-DA.jpg">
  <figcaption>Figure 1. Examples from the PACS dataset for DG. </figcaption>
</figure>

## 2. Tutorial Settings
In this series of tutorials, besides introducing and explaining outstanding DG methods intuitively, I also prepare those implementations and practice them on a real-world problem, which is classifying cardiac abnormalities from twelve-lead [ECGs](https://en.wikipedia.org/wiki/Electrocardiography) (see more details in [PhysioNet Challenge 2021](https://moody-challenge.physionet.org/2021/)). This can help you understand better as well as apply these methods to your own problems immediately. 
{: style="text-align: justify;"}

### Datasets
The datasets are from PhysioNet Challenge 2021, containing twelve-lead ECG recordings from 6 sources in 4 countries across 3 continents. Each recording was annotated with one or more of 26 types of cardiac abnormalities, which means the problem is multi-label classification. Figure 2 shows the number of data samples in each source domain, and Figure 3 illustrates the difference in the appearance of signals from 6 domains. 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/domain-generalization/data-sources.jpg">
  <figcaption>Figure 2. Number of data samples in each source domain. </figcaption>
</figure>
<figure class="align-left">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/domain-generalization/signal-appearance.jpg">
  <figcaption>Figure 3. Difference in the appearance of signals from 6 domains. </figcaption>
</figure>

I recommend you read some documents to understand what is ECG and cardiac abnormalities, as well as our problem: 
* [https://en.wikipedia.org/wiki/Electrocardiography](https://en.wikipedia.org/wiki/Electrocardiography)
* [https://www.who.int/health-topics/cardiovascular-diseases](https://www.who.int/health-topics/cardiovascular-diseases)
* [https://arxiv.org/abs/2207.12381](https://arxiv.org/abs/2207.12381)
{: style="text-align: justify;"}

Important things to remember: 
* `input`: 1D signal, a vector with shape with a length of 5000
* `output`: one or more of 26 classes, a vector of 26 elements, each 0 or 1
{: style="text-align: justify;"}

### Baseline