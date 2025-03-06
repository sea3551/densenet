import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torchvision.models import DenseNet161_Weights
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm  # 진행률 바 추가
from torchinfo import summary  # 모델 레이어 출력용

if __name__ == '__main__':
    # ✅ GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")

    # ✅ 데이터 로드
    data_dir = "spectrogram_images/day1"
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    print(f"🔹 클래스 목록: {dataset.classes}")

    # ✅ 모델 로드 및 수정
    model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(dataset.classes))
    model.to(device)

    # ✅ 모델 구조 출력
    summary(model, input_size=(1, dataset[0][0].shape[0], 400, 400), device=device)

    # ✅ 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # ✅ 모델 학습 시작
    print(f"\n🔹 Training Start | Epochs: {num_epochs}, Batch Size: {train_loader.batch_size}, Device: {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=running_loss / (total / 4), acc=correct / total)

        print(f"Epoch {epoch+1}/{num_epochs} ✅ Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

    # ✅ 성능 평가
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(model(images).argmax(1).cpu().numpy())

    # ✅ Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ✅ 평가 지표 출력
    print(f"\n🔹 Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"🔹 F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\n🔹 Classification Report:\n", classification_report(y_true, y_pred, target_names=dataset.classes))
