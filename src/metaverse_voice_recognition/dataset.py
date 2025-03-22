import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import random

class VoxCelebDataset(Dataset):
    """VoxCeleb 데이터셋을 위한 PyTorch Dataset 클래스"""
    
    def __init__(self, 
                 root_dir: str,
                 sample_rate: int = 16000,
                 duration: float = 3.0,
                 augment: bool = True):
        """
        Args:
            root_dir: VoxCeleb 데이터셋의 루트 디렉토리
            sample_rate: 목표 샘플링 레이트
            duration: 각 오디오 클립의 길이(초)
            augment: 데이터 증강 사용 여부
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment
        
        # 화자 ID와 파일 경로 수집
        self.speaker_files = self._collect_speaker_files()
        self.speakers = list(self.speaker_files.keys())
        
        # 데이터 증강을 위한 노이즈 파일 로드
        self.noise_files = []
        if self.augment:
            noise_dir = os.path.join(root_dir, 'noise')
            if os.path.exists(noise_dir):
                self.noise_files = [
                    os.path.join(noise_dir, f) 
                    for f in os.listdir(noise_dir) 
                    if f.endswith('.wav')
                ]
    
    def _collect_speaker_files(self) -> dict:
        """화자별 오디오 파일 경로를 수집합니다."""
        speaker_files = {}
        
        for speaker_id in os.listdir(self.root_dir):
            speaker_dir = os.path.join(self.root_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
                
            files = []
            for root, _, filenames in os.walk(speaker_dir):
                for filename in filenames:
                    if filename.endswith('.wav'):
                        files.append(os.path.join(root, filename))
            
            if files:
                speaker_files[speaker_id] = files
        
        return speaker_files
    
    def _load_audio(self, file_path: str) -> torch.Tensor:
        """오디오 파일을 로드하고 전처리합니다."""
        waveform, sr = torchaudio.load(file_path)
        
        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 리샘플링
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """오디오 데이터를 증강합니다."""
        if not self.augment or not self.noise_files:
            return waveform
            
        # 랜덤 노이즈 추가
        if random.random() < 0.5 and self.noise_files:
            noise_file = random.choice(self.noise_files)
            noise, _ = torchaudio.load(noise_file)
            if noise.shape[1] >= waveform.shape[1]:
                start = random.randint(0, noise.shape[1] - waveform.shape[1])
                noise = noise[:, start:start + waveform.shape[1]]
            else:
                noise = torch.nn.functional.pad(
                    noise, (0, waveform.shape[1] - noise.shape[1])
                )
            
            # SNR을 랜덤하게 설정 (5dB ~ 15dB)
            snr = random.uniform(5, 15)
            noise_scale = torch.norm(waveform) / (torch.norm(noise) * (10 ** (snr/20)))
            waveform = waveform + noise_scale * noise
        
        # 랜덤 볼륨 변경
        if random.random() < 0.5:
            scale = random.uniform(0.5, 2)
            waveform = waveform * scale
            
        return waveform
    
    def __len__(self) -> int:
        """데이터셋의 총 화자 수를 반환합니다."""
        return len(self.speakers)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        데이터셋에서 항목을 가져옵니다.
        
        Returns:
            anchor: 앵커 음성
            positive: 같은 화자의 다른 음성
            speaker_id: 화자 ID
        """
        speaker_id = self.speakers[idx]
        files = self.speaker_files[speaker_id]
        
        # 같은 화자의 서로 다른 두 파일 선택
        anchor_file, positive_file = random.sample(files, 2)
        
        # 오디오 로드 및 증강
        anchor = self._load_audio(anchor_file)
        anchor = self._augment_audio(anchor)
        
        positive = self._load_audio(positive_file)
        positive = self._augment_audio(positive)
        
        return anchor, positive, idx
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        배치의 데이터를 처리합니다.
        
        Args:
            batch: (anchor, positive, speaker_id) 튜플의 리스트
            
        Returns:
            anchors: 앵커 음성 배치
            positives: 양성 음성 배치
            speaker_ids: 화자 ID 배치
        """
        anchors, positives, speaker_ids = zip(*batch)
        
        # 패딩
        max_len = max(max(a.shape[1] for a in anchors), 
                     max(p.shape[1] for p in positives))
                     
        anchors_padded = torch.stack([
            torch.nn.functional.pad(a, (0, max_len - a.shape[1])) 
            for a in anchors
        ])
        
        positives_padded = torch.stack([
            torch.nn.functional.pad(p, (0, max_len - p.shape[1])) 
            for p in positives
        ])
        
        speaker_ids = torch.tensor(speaker_ids)
        
        return anchors_padded, positives_padded, speaker_ids 