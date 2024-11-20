import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 自定义数据集类以处理损坏的图像
class CustomDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        try:
            sample = Image.open(path).convert('RGB')
            sample = self.transform(sample)
            return sample, target
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None  # 返回 None 以跳过损坏的图像


# 自定义 collate_fn 跳过 None 值
def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]  # 过滤掉 None 值
    return list(zip(*batch)) if len(batch) > 0 else ([], [])


# 加载数据集
train_dataset = CustomDataset(
    root='D:/code/python/deeplearing/machine-learning/pre-train-model/kagglecatsanddogs_5340/train',
    transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

# 定义模型
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)  # 2类：猫和狗
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        if len(images) == 0:  # 如果没有有效图像，跳过此批次
            continue

        images, labels = torch.stack(images).to(device), torch.tensor(labels).to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 保存模型
torch.save(model.state_dict(), 'cat_dog_classifier.pth')


# 测试函数
def predict(image_path):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return 'dog' if predicted.item() == 1 else 'cat'

# 测试预测
# print(predict('path_to_your_test_image.jpg'))




