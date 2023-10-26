from torchvision.transforms import transforms

import data
import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


train_set = data.MyData(r'data\train',r'data\normal_data\train.csv', transform=transform)

data_train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

mymodel = model.ResNet()
mymodel.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mymodel.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)# 定义随机梯度下降优化器 )
epoch = 10
# for inputs, targets in data_train_loader:
#     pass
#     optimizer.zero_grad()
#     outs = mymodel(inputs)
#     loss = criterion(outs, targets)
#     loss.backward()
#     optimizer.step()

for i in range(epoch):
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):
        optimizer.zero_grad()
        outputs = mymodel(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predict = outputs.max(1)
        total += targets.size(0)
        _, realtag = targets.max(1)
        correct += predict.eq(realtag).sum().item()
        print(batch_idx, len(data_train_loader),'Loss: %.3f | (Acc: %.3f %%(%d/%d'%(train_loss/(batch_idx+1),100.*correct/total,correct,total))

    info = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "model": mymodel.state_dict()
    }

    torch.save(info, r"./model/model.pth")