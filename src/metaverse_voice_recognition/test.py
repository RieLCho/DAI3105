import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from metaverse_voice_recognition.dataset import VoxCelebDataset
from metaverse_voice_recognition.ecapa_tdnn import ECAPA_TDNN

def compute_eer(scores, labels):
    """Equal Error Rate를 계산합니다."""
    far = []  # False Acceptance Rate
    frr = []  # False Rejection Rate
    
    # 임계값을 변경하면서 FAR과 FRR 계산
    thresholds = np.linspace(np.min(scores), np.max(scores), 1000)
    for threshold in thresholds:
        far.append(np.sum((scores >= threshold) & (labels == 0)) / np.sum(labels == 0))
        frr.append(np.sum((scores < threshold) & (labels == 1)) / np.sum(labels == 1))
    
    far = np.array(far)
    frr = np.array(frr)
    
    # EER 계산
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    
    return eer

def evaluate_model(model_path: str, test_dir: str, device: str = 'cpu'):
    """학습된 모델을 평가합니다."""
    # 장치 설정
    device = torch.device(device)
    print(f'장치: {device}')
    
    # 모델 로드
    model = ECAPA_TDNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('모델을 로드했습니다.')
    
    # 테스트 데이터셋 준비
    test_dataset = VoxCelebDataset(test_dir, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=VoxCelebDataset.collate_fn
    )
    print(f'테스트 데이터셋 크기: {len(test_dataset)}')
    
    # 평가 지표를 위한 변수들
    all_scores = []
    all_labels = []
    correct = 0
    total = 0
    
    print('\n평가 시작...')
    with torch.no_grad():
        for batch_idx, (anchors, positives, labels) in enumerate(tqdm(test_loader)):
            anchors = anchors.to(device)
            positives = positives.to(device)
            
            # 임베딩 추출
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            
            # 코사인 유사도 계산
            anchor_embeddings = nn.functional.normalize(anchor_embeddings, p=2, dim=1)
            positive_embeddings = nn.functional.normalize(positive_embeddings, p=2, dim=1)
            
            similarity = torch.sum(anchor_embeddings * positive_embeddings, dim=1)
            
            # 점수와 레이블 저장
            all_scores.extend(similarity.cpu().numpy())
            all_labels.extend([1] * len(similarity))  # 모든 페어는 같은 화자
            
            # 정확도 계산 (임계값 0.5 사용)
            predictions = (similarity >= 0.5).float()
            total += len(predictions)
            correct += (predictions == 1).sum().item()
            
            # 다른 화자와의 비교 (negative pairs)
            for i in range(len(anchor_embeddings)):
                for j in range(len(positive_embeddings)):
                    if i != j:  # 다른 화자
                        similarity = torch.sum(anchor_embeddings[i] * positive_embeddings[j])
                        all_scores.append(similarity.cpu().item())
                        all_labels.append(0)  # 다른 화자
                        
                        # 정확도 계산
                        prediction = (similarity >= 0.5).float()
                        total += 1
                        correct += (prediction == 0).item()
    
    # 결과 계산
    accuracy = 100 * correct / total
    eer = compute_eer(np.array(all_scores), np.array(all_labels))
    
    print('\n평가 결과:')
    print(f'정확도: {accuracy:.2f}%')
    print(f'EER: {eer:.2f}%')
    
    return accuracy, eer

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ECAPA-TDNN model')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--test_dir', required=True, help='Test data directory')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_dir, args.device) 