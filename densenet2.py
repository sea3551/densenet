import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torchvision.models import DenseNet161_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np
import os
from tqdm import tqdm
from torchinfo import summary  # 모델 구조 출력용

# ✅ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# ✅ 데이터셋 로드 및 변환 정의
data_dir = "spectrogram_images/day1"
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# ✅ 클래스별 8:2 데이터 분할 함수
def stratified_split(dataset, train_ratio=0.8):
    labels = [label for _, label in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=42)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    return Subset(dataset, train_idx), Subset(dataset, test_idx)

# ✅ 데이터셋 분할
train_dataset, test_dataset = stratified_split(dataset, train_ratio=0.8)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"🔹 클래스 목록: {dataset.classes}")

# ✅ DenseNet 모델 정의
def build_model(num_classes):
    model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(device)

model = build_model(len(dataset.classes))

# ✅ 모델 구조 출력
summary(model, input_size=(1, dataset[0][0].shape[0], 400, 400), device=device)

# ✅ 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# ✅ 훈련 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0  # 최고 성능 저장을 위한 변수
    
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

        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} ✅ Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_acc:.4f}")

        # ✅ 최고 성능 모델 저장
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Best Model Saved with Accuracy: {best_acc:.4f}")

# ✅ Rank-K Accuracy 계산 함수
def rank_k_accuracy(y_true, y_scores, k=5):
    """
    Rank-K Accuracy를 계산합니다.
    :param y_true: 실제 정답 라벨 리스트
    :param y_scores: 모델의 예측 확률 (Softmax 값)
    :param k: Top-K 범위 (기본값: 5)
    :return: Rank-K 정확도
    """
    correct = 0
    for i in range(len(y_true)):
        top_k_predictions = np.argsort(y_scores[i])[-k:]  # 상위 K개 예측값 인덱스
        if y_true[i] in top_k_predictions:
            correct += 1
    return correct / len(y_true)

# ✅ 모델 평가 함수 (Confusion Matrix + Rank-K Accuracy 추가)
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())  # 확률 값 저장

    # ✅ Confusion Matrix 계산
    cm = confusion_matrix(y_true, y_pred)
    class_labels = dataset.classes

    # ✅ Confusion Matrix 출력 (터미널용)
    print("\n🔹 Confusion Matrix:")
    print("    " + "  ".join(f"{label[:4]:>4}" for label in class_labels))  # 클래스 라벨 출력
    for i, row in enumerate(cm):
        print(f"{class_labels[i][:4]:>4} " + "  ".join(f"{val:>4}" for val in row))

    # ✅ Rank-K Accuracy 계산
    rank1_acc = rank_k_accuracy(y_true, y_scores, k=1)
    rank3_acc = rank_k_accuracy(y_true, y_scores, k=3)
    rank5_acc = rank_k_accuracy(y_true, y_scores, k=5)

    # ✅ 평가 지표 출력
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\n🔹 Test Accuracy: {accuracy:.4f}")
    print(f"🔹 F1 Score: {f1:.4f}")
    print(f"🔹 Rank-1 Accuracy: {rank1_acc:.4f}")
    print(f"🔹 Rank-3 Accuracy: {rank3_acc:.4f}")
    print(f"🔹 Rank-5 Accuracy: {rank5_acc:.4f}")
    print("\n🔹 Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))

# ✅ 모델 학습 실행
train_model(model, train_loader, criterion, optimizer, num_epochs)

# ✅ 모델 평가 실행
evaluate_model(model, test_loader)
