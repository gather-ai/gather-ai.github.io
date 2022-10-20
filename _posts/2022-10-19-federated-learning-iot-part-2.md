---
title: "Federated Learning on IoT Devices Tutorials (Part 2): From Zero to Hero"
date: 2022-10-19
categories: 
  - Tutorials
tags: 
  - ai research
  - federated learning
  - iot
header: 
  image: "/assets/images/federated-learning-iot/flower-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page, this is part 2 of my tutorial series on deploying Federated Learning on IoT devices. In the [last article](https://gather-ai.github.io/tutorials/federated-learning-iot-part-1/), we discussed what FL is and built a network of IoT devices as well as environments for starting work. Today, I will guide you step by step to train a simple CNN model on the CIFAR10 dataset in real IoT devices by using [Flower](https://flower.dev/). Let's get started. 
{: style="text-align: justify;"}

## 1. Preparing Dataset

### CIFAR10 Dataset
The CIFAR10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Here are the classes in the dataset, as well as 10 random images from each: 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/federated-learning-iot/cifar10.jpg">
  <figcaption>Figure 1. CIFAR10 Dataset. Mount from [1]</figcaption>
</figure>

### Data Partitioning
In this tutorial, the training data are assigned to the clients in an IID setting. As mentioned before, our network has 10 clients in total, the training data is shuffled and uniformly divided into 10 partitions, each with 5000 images for each client. Note that each partition might be doesn't include 500 images for each class. 
{: style="text-align: justify;"}

After assigning data to clients, let's implement a Dataset class, which will be used in a PyTorch DataLoader. 
{: style="text-align: justify;"}

```python
"""
Snippet 1: Dataset class. 
"""
from libs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df, data_path, 
        image_size = (32, 32)
    ):
        self.df, self.data_path,  = df, data_path, 
        self.image_size = image_size

    def __len__(self, 
    ):
        return len(self.df)

    def __getitem__(self, 
        index
    ):
        row = self.df.iloc[index]

        image = np.load("{}/{}.npy".format(self.data_path, row["id"]))
        image = cv2.resize(image, self.image_size)/255
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)

        return torch.tensor(image).permute(2, 0, 1), row["label"]
```

## 2. Ingredients for Training

### A Simple CNN Model
For simplicity, I use a simple LeNet5 model, a pioneer CNN model, for deployment. Snippet 2 is an implementation of this model. 
{: style="text-align: justify;"}

```python
"""
Snippet 2: LeNet5 model. 
"""
from libs import *

class LeNet5(nn.Module):
    def __init__(self, 
        in_channels, num_classes
    ):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size = 5, stride = 1, padding = 0), 
            nn.BatchNorm2d(6), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2), 
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 0), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2), 
        )

        self.layer3 = nn.Sequential(
            nn.Linear(400, 120), 
            nn.ReLU(), 
            nn.Linear(120, 84), 
            nn.ReLU(), 
        )

        self.classifier = nn.Linear(84, num_classes)

    def forward(self, 
        input
    ):
        input = self.layer1(input)
        input = self.layer2(input)
        input = input.reshape(input.size(0), -1)

        input = self.layer3(input)

        logit = self.classifier(input)

        return logit
```

### A Training Function
We need a function that each client will use to perform training on their own data. All metrics during training should be logged and returned in a dictionary. 
{: style="text-align: justify;"}

```python
"""
Snippet 3: Training function. 
"""
from libs import *

def client_fit_fn(
    loaders, model, 
    num_epochs = 1, 
    device = torch.device("cpu"), 
    save_ckp_path = "./ckp.ptl", training_verbose = True
):
    print("\nStart Client Training ...\n" + " = "*16)
    model = model.to(device)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in tqdm(range(1, num_epochs + 1), disable = training_verbose):
        if training_verbose:print("epoch {:2}/{:2}".format(epoch, num_epochs) + "\n" + " - "*16)

        running_loss, running_correct = 0, 0
        for images, labels in tqdm(loaders["fit"], disable = not training_verbose):
            images, labels = images.float().to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss, running_correct = running_loss + loss.item()*images.size(0), running_correct + (torch.max(logits.data, 1)[1].detach().cpu() == labels.cpu()).sum().item()

        fit_loss, fit_accuracy,  = running_loss/len(loaders["fit"].dataset), running_correct/len(loaders["fit"].dataset), 
        if training_verbose:
            print("{:<5} - loss:{:.4f}, accuracy:{:.4f}".format(
                "fit", 
                fit_loss, fit_accuracy, 
            ))

        with torch.no_grad():
            model.eval()
            running_loss, running_correct = 0, 0
            for images, labels in tqdm(loaders["eval"], disable = not training_verbose):
                images, labels = images.float().to(device), labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                running_loss, running_correct = running_loss + loss.item()*images.size(0), running_correct + (torch.max(logits.data, 1)[1].detach().cpu() == labels.cpu()).sum().item()

        eval_loss, eval_accuracy,  = running_loss/len(loaders["eval"].dataset), running_correct/len(loaders["eval"].dataset), 
        if training_verbose:
            print("{:<5} - loss:{:.4f}, accuracy:{:.4f}".format(
                "eval", 
                eval_loss, eval_accuracy, 
            ))

    torch.save(model, save_ckp_path)
    print("\nFinish Client Training ...\n" + " = "*16)
    return {
        "fit_loss":fit_loss, "fit_accuracy":fit_accuracy, 
        "eval_loss":eval_loss, "eval_accuracy":eval_accuracy, 
    }
```

## 3. Server Site


## 4. Client Site


## References
[[1] CIFAR10 and CIFAR100 Datasets](https://www.cs.toronto.edu/~kriz/cifar.html)<br>
[[2] Flower: A Friendly Federated Learning Framework](https://flower.dev/)<br>