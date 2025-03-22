import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm

from metaverse_voice_recognition.dataset import VoxCelebDataset
from metaverse_voice_recognition.ecapa_tdnn import ECAPA_TDNN

class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax 손실 함수"""
    
    def __init__(self, embedding_dim: int, num_speakers: int, margin: float = 0.2, scale: float = 30):
        super(AAMSoftmax, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        nn.init.xavier_normal_(self.weight, gain=1)
        
        self.margin = margin
        self.scale = scale
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.th = np.cos(np.pi - margin)
        self.mm = np.sin(np.pi - margin) * margin
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: 화자 임베딩 (batch_size, embedding_dim)
            labels: 화자 레이블 (batch_size,)
        """
        # L2 정규화
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        weight = nn.functional.normalize(self.weight, p=2, dim=1)
        
        # 코사인 유사도
        cosine = nn.functional.linear(embeddings, weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # AAM 적용
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        # 원-핫 인코딩
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # 출력
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        
        # CrossEntropyLoss 적용
        loss = self.criterion(output, labels)
        
        return loss

def train_epoch(
    model: ECAPA_TDNN,
    train_loader: DataLoader,
    criterion: AAMSoftmax,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """한 에폭 동안 모델을 학습합니다."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (anchors, positives, labels) in enumerate(progress_bar):
        # 데이터를 GPU로 이동
        anchors = anchors.to(device)
        positives = positives.to(device)
        labels = labels.to(device)
        
        # 순전파
        optimizer.zero_grad()
        anchor_embeddings = model(anchors)
        positive_embeddings = model(positives)
        
        # 손실 계산
        anchor_loss = criterion(anchor_embeddings, labels)
        positive_loss = criterion(positive_embeddings, labels)
        loss = (anchor_loss + positive_loss) / 2
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        
        # 정확도 계산
        with torch.no_grad():
            anchor_output = nn.functional.linear(
                nn.functional.normalize(anchor_embeddings, p=2, dim=1),
                nn.functional.normalize(criterion.weight, p=2, dim=1)
            ) * criterion.scale
            positive_output = nn.functional.linear(
                nn.functional.normalize(positive_embeddings, p=2, dim=1),
                nn.functional.normalize(criterion.weight, p=2, dim=1)
            ) * criterion.scale
            
            _, anchor_predicted = torch.max(anchor_output, 1)
            _, positive_predicted = torch.max(positive_output, 1)
            total += labels.size(0) * 2
            correct += (anchor_predicted == labels).sum().item()
            correct += (positive_predicted == labels).sum().item()
        
        # 진행 상황 표시
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(
    model: ECAPA_TDNN,
    val_loader: DataLoader,
    criterion: AAMSoftmax,
    device: torch.device
) -> Tuple[float, float]:
    """검증 세트에서 모델을 평가합니다."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validating')
        for batch_idx, (anchors, positives, labels) in enumerate(progress_bar):
            anchors = anchors.to(device)
            positives = positives.to(device)
            labels = labels.to(device)
            
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            
            anchor_loss = criterion(anchor_embeddings, labels)
            positive_loss = criterion(positive_embeddings, labels)
            loss = (anchor_loss + positive_loss) / 2
            
            running_loss += loss.item()
            
            _, anchor_predicted = torch.max(anchor_loss.data, 1)
            _, positive_predicted = torch.max(positive_loss.data, 1)
            total += labels.size(0) * 2
            correct += (anchor_predicted == labels).sum().item()
            correct += (positive_predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def train_model(
    train_dir: str,
    val_dir: Optional[str] = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    save_dir: str = 'models'
) -> ECAPA_TDNN:
    """ECAPA-TDNN 모델을 학습합니다."""
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = VoxCelebDataset(train_dir, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=VoxCelebDataset.collate_fn
    )
    
    if val_dir:
        val_dataset = VoxCelebDataset(val_dir, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=VoxCelebDataset.collate_fn
        )
    
    # 모델, 손실 함수, 옵티마이저 초기화
    model = ECAPA_TDNN().to(device)
    criterion = AAMSoftmax(
        embedding_dim=192,
        num_speakers=len(train_dataset.speakers)
    ).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 체크포인트 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f'Training Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # 검증
        if val_dir:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
        
        # 학습률 조정
        scheduler.step()
        
        # 현재 모델 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
        }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
    
    return model

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ECAPA-TDNN model')
    parser.add_argument('--train_dir', required=True, help='Training data directory')
    parser.add_argument('--val_dir', help='Validation data directory')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    model = train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    ) 