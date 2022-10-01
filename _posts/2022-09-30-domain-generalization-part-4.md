---
title: "Domain Generalization Tutorials (Part 4): Domain Alignment (and More)"
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
Specifically, along the main task of cardiac abnormalities classification, DAT performs a subtask of domain identification and uses a gradient reversal layer to learn the representations in an adversarial manner. Figure 1 illustrates the architecture of the model and Snippet 1 describes the auxiliary module which performs DAT. 
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

Intuitively, the gradient reversal layer is skipped in the forward pass and just flips the sign of the gradient flow through it during the backpropagation process. Look at the position of this layer, it is placed right before the domain classifier $g_{d}$, this means that during training, $g_{d}$ is updated with $\frac{\partial L_{sub}}{\partial \theta_{g_d}}$ while the backbone $f$ is updated with $-\frac{\partial L_{sub}}{\partial \theta_{f}}$. In this way, the domain classifier learns how to use representations to identify the source domain of instances, but gives the reverse information to the backbone, forcing $f$ to generate domain-invariant representations. 
{: style="text-align: justify;"}

## 3. Instance-Batch Normalization Network

### Motivation
Nowadays, normalization layers are an important part of any neural network. There are many types of normalization techniques and each of them has its own characteristics and advantages, perhaps you have seen Figure 2 somewhere. We will talk about batch normalization (BN) and instance normalization (IN) here because of their effects on DG. 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/domain-generalization/normalization-techniques.jpg">
  <figcaption>Figure 2. Different normalization techniques. </figcaption>
</figure>

Although BN generally works well in a variety of tasks, it consistently degrades performance when it is trained in the presence of a large domain divergence. This is because the batch statistics overfit the particular training domains, resulting in poor generalization performance in unseen target domains. Meanwhile, IN does not depend on batch statistics. This property allows the network to learn feature representations that less overfit a particular domain. The downside of IN, however, is that it makes the features less discriminative with respect to instance categories, which is guaranteed in BN in contrast. Instance-Batch normalization (I-BN) is a mixture of BN and IN, which is introduced to reap the benefits of IN of learning domain-invariant representations while maintaining the ability to capture discriminative representations from BN. 
{: style="text-align: justify;"}

### Method
Snippet 3 is a simple implementation of an I-BN layer, just half of BN and half of IN. 
{: style="text-align: justify;"}

```python
"""
Snippet 3: I-BN layer. 
"""
import torch
import torch.nn as nn

class InstanceBatchNorm1d(nn.Module):
  def __init__(self, planes):
    super(InstanceBatchNorm1d, self).__init__()
    self.half_planes = planes//2

    self.BN, self.IN = nn.BatchNorm1d(planes - self.half_planes), nn.InstanceNorm1d(self.half_planes, affine = True)

  def forward(self, input):
    half_input = torch.split(input, self.half_planes, dim = 1)

    half_BN, half_IN = self.BN(half_input[0].contiguous()), self.IN(half_input[1].contiguous())
    return torch.cat((half_BN, half_IN), dim = 1)
```

## 4. Domain-Specific Optimized Normalization Network

### Motivation

### Method

## 5. Results
The table below shows the performance of the two presented methods in this article. 
{: style="text-align: justify;"}

To be continued ...
{: style="text-align: justify;"}

## References
[[1] Domain Generalization: A Survey](https://arxiv.org/abs/2103.02503)<br>
[[2] Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)<br>
[[3] Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net](https://arxiv.org/abs/1807.09441)<br>
[[4] Learning to Optimize Domain Specific Normalization for Domain Generalization](https://arxiv.org/abs/1907.04275)<br>
{: style="text-align: justify;"}