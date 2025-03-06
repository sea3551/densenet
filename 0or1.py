import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ 1. 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# ✅ 2. 이미지 변환 (전처리)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ 3. 데이터셋 로드
data_dir = "spectrogram_images/day1"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# ✅ 4. 사용자별 이미지 그룹화
person_to_images = {}
for img_path, label in dataset.samples:
    person = dataset.classes[label]
    if person not in person_to_images:
        person_to_images[person] = []
    person_to_images[person].append(img_path)

# ✅ 5. 동일인/비교군 데이터셋 구성
class PairwiseDataset(Dataset):
    def __init__(self, person_to_images, transform=None):
        self.transform = transform
        self.pairs = []
        self.labels = []
        
        persons = list(person_to_images.keys())

        for person in persons:
            images = person_to_images[person]
            # 동일한 사람(POSITIVE PAIR)
            for i in range(len(images) - 1):
                for j in range(i + 1, len(images)):
                    self.pairs.append((images[i], images[j]))
                    self.labels.append(1)  # 동일인

            # 다른 사람(NEGATIVE PAIR)
            for _ in range(len(images)):  # 동일한 수만큼 생성
                other_person = random.choice([p for p in persons if p != person])
                other_image = random.choice(person_to_images[other_person])
                self.pairs.append((images[0], other_image))
                self.labels.append(0)  # 다른 사람
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        img1 = dataset.loader(img1_path)
        img2 = dataset.loader(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# ✅ 6. 데이터셋 생성
pairwise_dataset = PairwiseDataset(person_to_images, transform=transform)
train_size = int(0.8 * len(pairwise_dataset))
test_size = len(pairwise_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(pairwise_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ✅ 7. DenseNet 이진 분류 모델 정의
class DenseNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(DenseNetBinaryClassifier, self).__init__()
        self.base_model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, 512)
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 이진 분류
        )

    def forward(self, img1, img2):
        feat1 = self.base_model(img1)
        feat2 = self.base_model(img2)
        combined = torch.cat((feat1, feat2), dim=1)  # Feature Concatenation
        output = self.fc(combined)
        return output

# ✅ 8. 모델 초기화
model = DenseNetBinaryClassifier().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# ✅ 9. 모델 학습
print(f"\n🔹 Training Start | Epochs: {num_epochs}, Batch Size: {train_loader.batch_size}, Device: {device}")

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for img1, img2, labels in progress_bar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=loss.item(), acc=correct / total)

    print(f"Epoch {epoch+1}/{num_epochs} ✅ Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

# ✅ 10. 모델 평가
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for img1, img2, labels in test_loader:
        img1, img2, labels = img1.to(device), img2.to(device)
        outputs = model(img1, img2).squeeze()
        predicted = (outputs > 0.5).float()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# ✅ 11. 혼동 행렬 및 성능 출력
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Different", "Same"], yticklabels=["Different", "Same"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(f"\n🔹 Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"🔹 F1 Score: {f1_score(y_true, y_pred):.4f}")
print("\n🔹 Classification Report:\n", classification_report(y_true, y_pred, target_names=["Different", "Same"]))
