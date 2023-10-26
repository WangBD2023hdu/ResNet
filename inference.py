from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import model
import data
import torch
import torch.nn as nn
model_path = "./model/model.pth" #
save_info = torch.load(model_path)
model = model.ResNet()
criterion = nn.CrossEntropyLoss()
model.load_state_dict(save_info["model"])
model.eval()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


train_set = data.MyData(r'data\test',r'data\normal_data\test.csv', transform=transform)


data_test_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=0, drop_last=True)

test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_test_loader):
        pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predict = outputs.max(1)
        total += targets.size(0)
        _, realtag = targets.max(1)
        correct += predict.eq(realtag).sum().item()
        print(batch_idx, len(data_test_loader), 'Loss: %.3f | (Acc: %.3f %%(%d/%d' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
