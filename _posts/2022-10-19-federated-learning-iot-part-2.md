---
title: "Federated Learning on IoT Devices - Part 2"
subtitle: "From Zero to Hero"
date: 2022-10-19
categories: 
  - Tutorials
tags: 
  - AI Research
  - Federated Learning
  - IoT
header: 
  image: "/assets/images/federated-learning-iot/flower-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page, this is part 2 of my tutorial series on deploying Federated Learning on IoT devices. In the [last article](https://gather-ai.github.io/tutorials/federated-learning-iot-part-1/), we discussed what FL is and built a network of IoT devices as well as environments for starting work. Today, I will guide you step by step to train a simple CNN model on the CIFAR10 dataset in real IoT devices by using [Flower](https://flower.dev/). Let's get started. 

## 1. Preparing Dataset

### CIFAR10 Dataset
The CIFAR10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Here are the classes in the dataset, as well as 10 random images from each: 

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/federated-learning-iot/cifar10.jpg">
  <figcaption>Figure 1. CIFAR10 Dataset. Mount from [1]</figcaption>
</figure>

### Data Partitioning
In this tutorial, the training data are assigned to the clients in an IID setting. As mentioned before, our network has 10 clients in total, the training data is shuffled and uniformly divided into 10 partitions, each with 5000 images for each client. Note that each partition might be doesn't include 500 images for each class. 

After assigning data to clients, let's implement a Dataset class, which will be used in a PyTorch DataLoader. 

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
We can use our laptop to work as a server, at each round, the server sent a global model to all clients to perform on-device training. When clients finish their training, they will send their local models back to the server, then the global model is updated by an FL strategy, FedAvg for example, where the server averages all models from clients and start the next round. 

We will modify the `FedAvg` class of Flower to save the global at each round. 
```python
"""
Snippet 4: FedAvg strategy. 
"""
from libs import *

def metrics_aggregation_fn(metrics):
    fit_losses, fit_accuracies,  = [metric["fit_loss"] for _, metric in metrics], [metric["fit_accuracy"] for _, metric in metrics], 
    eval_losses, eval_accuracies,  = [metric["eval_loss"] for _, metric in metrics], [metric["eval_accuracy"] for _, metric in metrics], 
    aggregated_metrics = {
        "fit_loss":sum(fit_losses)/len(fit_losses), "fit_accuracy":sum(fit_accuracies)/len(fit_accuracies), 
        "eval_loss":sum(eval_losses)/len(eval_losses), "eval_accuracy":sum(eval_accuracies)/len(eval_accuracies), 
    }

    return aggregated_metrics

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        initial_model, 
        save_ckp_path, 
        *args, **kwargs
    ):
        self.initial_model = initial_model
        self.save_ckp_path = save_ckp_path
        super().__init__(*args, **kwargs)
    def aggregate_fit(self, 
        server_round, 
        results, failures
    ):
        aggregated_metrics = metrics_aggregation_fn([(result.num_examples, result.metrics) for _, result in results])
        wandb.log({"fit_loss":aggregated_metrics["fit_loss"]}, step = server_round), wandb.log({"fit_accuracy":aggregated_metrics["fit_accuracy"]}, step = server_round), 
        wandb.log({"eval_loss":aggregated_metrics["eval_loss"]}, step = server_round), wandb.log({"eval_accuracy":aggregated_metrics["eval_accuracy"]}, step = server_round), 

        aggregated_parameters, results = super().aggregate_fit(
            server_round, 
            results, failures
        )
        if aggregated_parameters is not None:
            self.initial_model.load_state_dict(OrderedDict({key:torch.tensor(value) for key, value in zip(self.initial_model.state_dict().keys(), fl.common.parameters_to_weights(aggregated_parameters))}), strict = True)
            torch.save(self.initial_model, self.save_ckp_path)

        return aggregated_parameters, {}
```

The server can be easily started by passing your laptop IP address and an arbitrary port into the `start_server` function. 

```python
"""
Snippet 5: Server site. 
"""
from libs import *

from data import ImageDataset
from nets import LeNet5
from strategies import FedAvg
from engines import server_test_fn

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type = str, default = "192.168.50.102"), parser.add_argument("--server_port", type = int)
parser.add_argument("--dataset", type = str, default = "CIFAR10"), parser.add_argument("--num_clients", type = int, default = 10)
parser.add_argument("--num_rounds", type = int, default = 100)
args = parser.parse_args()

wandb.login()
wandb.init(project = "FL-IoT", name = "{}".format(args.dataset))

initial_model = LeNet5(1 if "MNIST" in args.dataset else 3, num_classes = 10)
initial_parameters = [value.cpu().numpy() for key, value in initial_model.state_dict().items()]
save_ckp_path = "../ckps/{}/server.ptl".format(args.dataset)
if not os.path.exists("/".join(save_ckp_path.split("/")[:-1])):
    os.makedirs("/".join(save_ckp_path.split("/")[:-1]))
fl.server.start_server(
    server_address = "{}:{}".format(args.server_address, args.server_port), 
    config = {"num_rounds":args.num_rounds}, 
    strategy = FedAvg(min_available_clients = args.num_clients, 
        min_fit_clients = args.num_clients, 
        min_eval_clients = args.num_clients, 
        initial_parameters = fl.common.weights_to_parameters(initial_parameters), 
        initial_model = initial_model, 
        save_ckp_path = save_ckp_path, 
    )
)
```

## 4. Client Site
For the client, we need to create a `Client` class that inherits from Flower’s `Client` and contains 4 methods `get_parameters`, `set_parameters`, `fit`, and `evaluate`. Then, pass the server’s IP address and its opened port, the rest is similar to traditional ML projects. 

```python
"""
Snippet 6: Client site. 
"""
from libs import *

from data import ImageDataset
from nets import LeNet5
from engines import client_fit_fn

class Client(fl.client.NumPyClient):
    def __init__(self, 
        loaders, model, 
        num_epochs = 1, 
        device = torch.device("cpu"), 
        save_ckp_path = "./ckp.ptl", training_verbose = True
    ):
        self.loaders, self.model,  = loaders, model, 
        self.num_epochs = num_epochs
        self.device = device
        self.save_ckp_path, self.training_verbose = save_ckp_path, training_verbose

        self.model = self.model.to(device)

    def get_parameters(self, 
        config
    ):
        self.model.train()
        return [value.cpu().numpy() for key, value in self.model.state_dict().items()]

    def set_parameters(self, 
        parameters, 
    ):
        self.model.train()
        self.model.load_state_dict(OrderedDict({key:torch.tensor(value) for key, value in zip(self.model.state_dict().keys(), parameters)}), strict = True)
    def fit(self, 
        parameters, config
    ):
        self.set_parameters(parameters)
        self.model.train()
        history = client_fit_fn(
            self.loaders, self.model, 
            self.num_epochs, 
            self.device, 
            self.save_ckp_path, self.training_verbose
        )
        return self.get_parameters(config = {}), len(loaders["fit"].dataset), history
    def evaluate(self, 
        parameters, config
    ):
        return float(len(loaders["eval"].dataset)), len(loaders["eval"].dataset), {}

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type = str, default = "192.168.50.102"), parser.add_argument("--server_port", type = int)
parser.add_argument("--dataset", type = str, default = "CIFAR10"), parser.add_argument("--cid", type = int)
args = parser.parse_args()

df = pandas.read_csv("../datasets/{}/clients/client_{}.csv".format(args.dataset, args.cid))
loaders = {
    "fit":torch.utils.data.DataLoader(
        ImageDataset(
            df = df[df["phase"] == "fit"], data_path = "../datasets/{}/train".format(args.dataset), 
        ), batch_size = 32, 
        shuffle = True
    ), 
    "eval":torch.utils.data.DataLoader(
        ImageDataset(
            df = df[df["phase"] == "eval"], data_path = "../datasets/{}/train".format(args.dataset), 
        ), batch_size = 32*2, 
        shuffle = False
    ), 
}
model = LeNet5(1 if "MNIST" in args.dataset else 3, num_classes = 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_ckp_path = "../ckps/{}/client_{}.ptl".format(args.dataset, args.cid)
if not os.path.exists("/".join(save_ckp_path.split("/")[:-1])):
    os.makedirs("/".join(save_ckp_path.split("/")[:-1]))
client = Client(
    loaders, model, 
    num_epochs = 1, 
    device = device, 
    save_ckp_path = save_ckp_path, training_verbose = True
)
fl.client.start_numpy_client(
    server_address = "{}:{}".format(args.server_address, args.server_port), 
    client = client, 
)
```

Now, everything is ready for starting. On your laptop, run the server, and on each device, run the client. As you can see, I use `wandb` to log all metrics during training. This is what they look like after 100 rounds: 

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/federated-learning-iot/metrics.jpg">
  <figcaption>Figure 2. Training Loss and Accuracy. </figcaption>
</figure>

Stay tuned for more content ...

## References
[[1] CIFAR10 and CIFAR100 Datasets](https://www.cs.toronto.edu/~kriz/cifar.html)<br>
[[2] Flower: A Friendly Federated Learning Framework](https://flower.dev/)<br>
{: style="font-size: 12px;"}