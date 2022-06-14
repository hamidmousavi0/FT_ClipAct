import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy
from Data_load import load_Cifar10
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        model = model.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = torch.softmax(logits, dim=1)
        return logits,probas
def train_eval_alexnet(NUM_EPOCHS,train_loader,valid_loader,training):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlexNet(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if training:
        for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
            print(epoch)
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                model = model.to(DEVICE)
        #
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                logits,probas= model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print(compute_accuracy(model,valid_loader,'cpu'))
        path = "./alexnet.pth"
        torch.save(model,path)
        print('Finished Training')
    else:
        path = "./alexnet.pth"
        model = torch.load(path)

    return model, compute_accuracy(model,valid_loader,DEVICE)
def train(train_loader, model, criterion, optimizer, device,epochs):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    for epoch in range(0, epochs):
        print(epoch)
        for X, y_true in train_loader:
            optimizer.zero_grad()

            X = X.to(device)
            model = model.to(device)
            y_true = y_true.to(device)

            # Forward pass
            y_hat, _ = model(X)
            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * X.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()
    return  model

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss
if __name__ == '__main__':
    train_loader,valid_loader = load_Cifar10()
    model,acc = train_eval_alexnet(20,train_loader,valid_loader)