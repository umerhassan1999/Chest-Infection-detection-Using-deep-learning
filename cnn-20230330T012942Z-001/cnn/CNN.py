
from __future__ import division
import torch
from numpy import *
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0)



transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='train',transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='test',transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True, num_workers=0)

classes = ('Covid', 'Normal', 'Viral_Pneumonia')
# =============================================================================


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) 
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, 6)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # output size: [batch_size, 32, 255, 255]
        x = self.pool(F.relu(self.conv2(x)))  # output size: [batch_size, 64, 126, 126]

        x = x.view(-1, 64 * 14 * 14)  # output size: [batch_size, 64*126*126]
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        x = self.softmax(x) #confrim it.


        return x


x = torch.randn(4, 3, 64, 64)  # (batch size or #of images,channels RGB,width,height)
model = CNN()
output = model(x)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)#Implements Adam algorithm.



trainloss = []
testloss = []
trainaccuracy = []
testaccuracy = []
epoch = 60


for epoch in range(epoch):
    correct1 = 0
    itr1 = 0
    itrloss = 0
    model.train()
    epoch =epoch+1
    epoch=str(epoch)
    print("Epoch "+ epoch +" ::")


    for i, (images, labels) in enumerate(trainloader):

        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor
        outputs = model(images)
        # print(outputs.data.cpu().numpy())
        loss = criterion(outputs, labels)
        itrloss += loss.item()
        loss.backward()  # once the gradients are computed using e.g.
        optimizer.step()  # method, that updates the parameters
        _, predicted = torch.max(outputs, 1)
        correct1 += (predicted == labels).sum().numpy()
        itr1 += 1
        #print(itr1,  "/200")

    trainloss.append(itrloss / itr1)

    trainaccuracy.append((100 * correct1) / len(trainset))
    print('training loss:%2f %%' % (itrloss / itr1))
    print('training accuracy:%2f %%' % (100 * correct1 / len(trainset)))
    torch.save({'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),}, 'xrayClassifier.pth')  #save .pth model somewhere


    # testing
    itr2loss = 0.0
    correct = 0
    total = 0.0
    itr2 = 0

    model.eval()
    for images, labels in testloader:

        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        loss = criterion(outputs, labels)
        itr2loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().numpy()
        itr2 += 1

    testloss.append(itr2loss / itr2)
    testaccuracy.append(100 * correct / len(testset))
    print('test loss:%4f %%' % (loss / itr2))
    print('test accuracy:%4f %%' % (100 * correct / len(testset)))
    print("=========================================================")

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))



class_correct = list(0 for i in range(6))
class_total = list(0 for i in range(6))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(len(classes)):
     print('Accuracy of %5s : %2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# print (trainloss)
# print (testloss)
# print (trainaccuracy)
# print (testaccuracy)

plt.plot(trainloss, label='training loss')
plt.plot(testloss, label='testing loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['trainloss', 'testloss'], loc='upper left')
plt.show()
#
plt.plot(trainaccuracy, label='training accuracy')
plt.plot(testaccuracy, label='testing accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training accuracy', 'testing accuracy'], loc='upper left')
plt.show()

plt.figure(figsize=(4, 4))
plt.plot(trainloss,label='training loss')
plt.plot(testloss,label='testing loss')
plt.plot(trainaccuracy, label='training accuracy')
plt.plot(testaccuracy, label='testing accuracy')
plt.legend()
plt.show()

