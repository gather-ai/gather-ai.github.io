---
title: "Domain Generalization Tutorials (Part 3): Inter-domain Data Augmentation"
date: 2022-09-25
categories: 
  - Tutorials
tags: 
  - deep learning
header: 
  image: "/assets/images/domain-generalization/shift-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page, this is part 3 of my tutorial series about the topic of Domain Generalization (DG). From this article, we will explore domain-aware approaches which take the problem domain shift into account. Today, I introduce the first family of methods which is **inter-domain data augmentation**. 
{: style="text-align: justify;"}

You can find the source code of the whole series [here](https://github.com/lhkhiem28/DGECG). 
{: style="text-align: justify;"}
{: .notice--info}

## 1. Inter-domain Data Augmentation
**Mixing data augmentation** is an emerging type of augmentation method that has shown superior in recent years. The methods of this type do a [convex combination](https://en.wikipedia.org/wiki/Convex_combination) (mix) on two data instances at the input or feature level, hence generating a new instance for training ML models. Differing from conventional data augmentation such as crop, scale, or cut out which preserves the context of the original instance, mixing augmentation creates instances with new contexts, in other words, new domains, this is extremely suitable for solving the DG problem. Because these methods perfectly fit into mini-batch training, we have two ways to select data instances for doing the mixing, without domain labels (random mixing) and with domain labels (inter-domain mixing). Figure 1 illustrates these selection strategies. Let’s dive into Mixup and MixStyle, the two most popular and effective augmentation methods for DG. 
{: style="text-align: justify;"}

## References
[[1] ]()<br>
{: style="text-align: justify;"}