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
    nn.Linear(512, num_domains), 
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
Snippet 3 is a simple implementation of a one-dimensional I-BN layer, just half of BN and half of IN. It is straightforward to extend the implementation to higher-dimension usages. 
{: style="text-align: justify;"}

```python
"""
Snippet 3: I-BN layer. 
"""
import torch
import torch.nn as nn

class Instance_BatchNorm1d(nn.Module):
  def __init__(self, num_features):
    super(Instance_BatchNorm1d, self).__init__()

    self.half_num_features = num_features//2
    self.BN, self.IN = nn.BatchNorm1d(num_features - self.half_num_features), nn.InstanceNorm1d(self.half_num_features, affine = True)

  def forward(self, input):
    half_input = torch.split(input, self.half_num_features, dim = 1)
    half_BN, half_IN = self.BN(half_input[0].contiguous()), self.IN(half_input[1].contiguous())

    return torch.cat((half_BN, half_IN), dim = 1)
```

But where to place I-BN layers in a specific network, a ResNet-like model for example? Another observation showed that, for BN-based CNNs, the feature divergence caused by appearance variance (domain shift) mainly lies in the shallow half of the CNN, while the feature discrimination for categories is high in deep layers, but also exists in shallow layers. Therefore, an original ResNet can is modified as follows to become an I-BN ResNet: 
{: style="text-align: justify;"}
* Only use I-BN layers in the first three residual blocks and leave the fourth block as normal (similar to MixStyle in the [previous article](https://gather-ai.github.io/tutorials/domain-generalization-part-3/))
* For each selected block, only replace the BN layer after the first convolution layer in the main path with an I-BN layer
{: style="text-align: justify;"}

Snippet 4 illustrates this setting. 
{: style="text-align: justify;"}

```python
"""
Snippet 4: I-BN ResNet setting. 
"""
import torch.nn as nn

class SEResNet34(nn.Module):

  ...
  self.block = I_NBSEBlock()
  ...

  ...
  self.stem = ...
  self.stage_0 = nn.Sequential(
    self.block(i_bn = True), 
    self.block(i_bn = True), 
    self.block(i_bn = True), 
  )

  self.stage_1 = nn.Sequential(
    self.block(i_bn = True), 
    self.block(i_bn = True), 
    self.block(i_bn = True), 
    self.block(i_bn = True), 
  )
  self.stage_2 = nn.Sequential(
    self.block(i_bn = True), 
    self.block(i_bn = True), 
    self.block(i_bn = True), 
    self.block(i_bn = True), 
    self.block(i_bn = True), 
    self.block(i_bn = True), 
  )
  self.stage_3 = nn.Sequential(
    self.block(i_bn = False), 
    self.block(i_bn = False), 
    self.block(i_bn = False), 
  )
  ...
```

## 4. Domain-Specific I-BN Network

### Motivation
Domain alignment methods generally have a common limitation, which will be discussed and addressed here. Look back to an illustration of DG from part 1, where a classifier trained in _sketch_, _cartoon_, _art painting_ images encounters instances from a novel domain _photo_ at test-time. 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/domain-generalization/DG-DA.jpg">
  <figcaption>Figure 3. Examples from the PACS dataset for DG. Adapted from [1]. </figcaption>
</figure>

It is reasonable to note that leveraging the relative similarity of the _photo_ instances to instances from _art painting_ might result in better predictions compared to a setting where the model relies solely on invariant characteristics across domains. Both covered methods try to learn domain-invariant representations while ignoring domain-specific features, features that are specific to individual domains. 
{: style="text-align: justify;"}

Extending from the above I-BN Net, domain-specific I-BN Net (DS I-BN Net) is developed which aims to capture both domain-invariant and domain-specific features from multi-source domain data. 
{: style="text-align: justify;"}

### Method
In particular, an original ResNet can is modified to become a DS I-BN ResNet in the following two steps: 
{: style="text-align: justify;"}
* Turn all BN layers in the model into (domain-specific BN) [DSBN](https://arxiv.org/abs/1906.03950) modules
* Replace BN layers with I-BN layers at the same positions as I-BN ResNet
{: style="text-align: justify;"}

What is the DSBN? DSBN is a module that consists of $M$ BN layers, using parameters of each BN layer to capture domain-specific features of each individual domain in $M$ source domains. Specifically, during training, instances from domain $m$, $\mathbf{X}_{m}$ only go through the $m^{th}$ BN layer in the DSBN module. Figure 4 illustrates the module and Snippet 5 is its implementation in a 1-dimensional version. 
{: style="text-align: justify;"}

<figure class="align-center" style="width: 500px">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/domain-generalization/domain-specific-batch-normalization.jpg">
  <figcaption>Figure 4. Domain-specific batch normalization module architecture. </figcaption>
</figure>

```python
"""
Snippet 5: I-BN layer. 
"""
import torch
import torch.nn as nn

class DomainSpecificBatchNorm1d(nn.Module):
  def __init__(self, num_features, num_domains):
    super(DomainSpecificBatchNorm1d, self).__init__()

    self.num_domains = num_domains
    self.BNs = nn.ModuleList(
      [nn.BatchNorm1d(num_features) for _ in range(self.num_domains)]
    )

  def forward(self, input, domains, is_training = True, running_domain = None):
    domain_uniques = torch.unique(domains)

    if is_training:
      outputs = [self.BNs[i](input[domains == domain_uniques[i]]) for i in range(domain_uniques.shape[0])]
      return torch.concat(outputs)
    else:
      output = self.BNs[running_domain](output)
      return output
```

At inference time, a test instance is fed into all $M$ “sub-networks” of all domains to get $M$ logits. The final logit is averaged over these $M$ logits and made the prediction. 
{: style="text-align: justify;"}

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
[[5] Learning to Balance Specificity and Invariance for In and Out of Domain Generalization](https://arxiv.org/abs/2008.12839)<br>