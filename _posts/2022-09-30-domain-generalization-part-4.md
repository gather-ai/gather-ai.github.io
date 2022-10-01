---
title: "Domain Generalization Tutorials (Part 4): Domain Alignment"
date: 2022-09-30
categories: 
  - Tutorials
tags: 
  - ai research
  - deep learning
header: 
  image: "/assets/images/domain-generalization/shift-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page, this is part 4 of my tutorial series about the topic of Domain Generalization (DG). This article will cover the approach of **domain alignment**, to which most existing DG methods belong. 
{: style="text-align: justify;"}

You can find the source code of the whole series [here](https://github.com/lhkhiem28/DGECG). 
{: style="text-align: justify;"}
{: .notice--info}

## 1. Domain Alignment
The central idea of domain alignment is to minimize the difference among source domains for learning _domain-invariant representations_. The motivation is straightforward: features that are invariant to the source domains should also generalize well on any unseen target domain. Traditionally, the difference among source domains is modeled by Feature Correlation or Maximum Mean Discrepancy, these entities are minimized to learn domain-invariant representations. However, let’s explore simpler and more effective domain alignment methods. 
{: style="text-align: justify;"}

## 2. Domain-Adversarial Training

## 3. Instance-Batch Normalization

## 4. Domain-Specific Optimized Normalization

## 5. Results
The table below shows the performance of the two presented methods in this article. 
{: style="text-align: justify;"}

To be continued ...
{: style="text-align: justify;"}

## References
[[1] Domain Generalization: A Survey](https://arxiv.org/abs/2103.02503)<br>
{: style="text-align: justify;"}