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
The central idea of domain alignment is to minimize the difference among source domains for learning _domain-invariant representations_. The motivation is straightforward: features that are invariant to the source domains should also generalize well on any unseen target domain. Traditionally, the difference among source domains is modeled by [Feature Correlation](https://arxiv.org/abs/1612.01939) or [Maximum Mean Discrepancy](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html), these entities are minimized to learn domain-invariant representations. However, let's explore simpler and more effective domain alignment methods. 
{: style="text-align: justify;"}

## 2. Domain-Adversarial Training

### Motivation
Don't be afraid to see the word "adversarial", this method is simple to understand if you have read about multi-task learning in [part 2](https://gather-ai.github.io/tutorials/domain-generalization-part-2/) of the series, but if not, it's still simple. Domain-adversarial training (DAT) perfectly represents the spirit of the domain alignment approach, that is to learn the feature cannot tell which source domain the instance came from. 
{: style="text-align: justify;"}

By leveraging a multi-task learning setting, DAT combines discriminativeness and domain-invariance into the same representations. To this end, a subtle trick is introduced along with the main method. 
{: style="text-align: justify;"}

### Method
Specifically, along the main task of cardiac abnormalities classification, DAT performs a subtask of domain identification and uses a gradient reversal layer to learn the representation in an adversarial manner. Figure 1 illustrates the architecture of the model and Snippet 1 describes the auxiliary module which performs DAT. 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/domain-generalization/domain-adversarial-training.jpg">
  <figcaption>Figure 1. Domain-adversarial training architecture. </figcaption>
</figure>

```python
"""
Snippet 1: DAT module. 
"""
import torch.nn as nn

class SEResNet34(nn.Module):

  ...
  self.auxiliary = nn.Sequential(
    GradientReversal(), 
    nn.Dropout(0.2), 
    nn.Linear(512, 5), 
  )
  ...

  def forward(self, inputs):
    ...
    return self.classifier(feature), self.auxiliary(feature)
```

The model is optimized with a combined loss similar to multi-task learning. Snippet 2 describes the optimization process. 
{: style="text-align: justify;"}

```python
"""
Snippet 2: Optimization process. 
"""
import torch.nn.functional as F

...
logits, sub_logits = model(ecgs)
loss, sub_loss = F.binary_cross_entropy_with_logits(logits, labels), F.cross_entropy(sub_logits, domains)
(loss + auxiliary_lambda*sub_loss).backward()
...
```

## 3. Instance-Batch Normalization

## 4. Domain-Specific Optimized Normalization

## 5. Results
The table below shows the performance of the two presented methods in this article. 
{: style="text-align: justify;"}

To be continued ...
{: style="text-align: justify;"}

## References
[[1] Domain Generalization: A Survey](https://arxiv.org/abs/2103.02503)<br>
[[2] Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)<br>
{: style="text-align: justify;"}