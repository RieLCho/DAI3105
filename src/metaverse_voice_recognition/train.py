import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torchaudio

from metaverse_voice_recognition.dataset import VoxCelebDataset
from metaverse_voice_recognition.ecapa_tdnn import ECAPA_TDNN
from metaverse_voice_recognition.config import ModelConfig, TrainingConfig
from .model import SpeakerNet

def count_parameters(model: nn.Module) -> int:
    """모델의 학습 가능한 파라미터 수를 반환합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> Tuple[float, float]:
    """한 에폭 동안 모델을 학습합니다."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f'Epoch {epoch}/{total_epochs} [Train]',
        leave=False
    )
    
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
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> Tuple[float, float]:
    """검증 세트에서 모델을 평가합니다."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(
            val_loader,
            desc=f'Epoch {epoch}/{total_epochs} [Valid]',
            leave=False
        )
        
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
            
            # 정확도 계산
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
            
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.files = []
        self.labels = {}
        label_idx = 0
        
        # 데이터 디렉토리 탐색
        for speaker_dir in self.data_dir.iterdir():
            if speaker_dir.is_dir():
                self.labels[speaker_dir.name] = label_idx
                for wav_file in speaker_dir.glob("*.wav"):
                    self.files.append(wav_file)
                label_idx += 1
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        label = self.labels[file_path.parent.name]
        return waveform, label

def train_model(config: TrainingConfig) -> ECAPA_TDNN:
    """ECAPA-TDNN 모델을 학습합니다."""
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = AudioDataset(config.train_dir)
    val_dataset = AudioDataset(config.val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 모델 초기화
    model = SpeakerNet(num_classes=len(train_dataset.labels)).to(device)
    print(f'모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')
    
    # 손실 함수, 옵티마이저 초기화
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Early stopping 설정
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("\n학습 시작...")
    
    for epoch in range(config.num_epochs):
        print(f"\n\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # 학습
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 검증
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"Training Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 모델 저장
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pth'))
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{patience}번 동안 성능 향상이 없어 학습을 조기 종료합니다.")
                break
    
    print("\n학습 완료!")
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
    
    config = TrainingConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    model = train_model(config) 