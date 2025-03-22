import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class SpeakerNet(nn.Module):
    def __init__(self, num_classes=40):
        super(SpeakerNet, self).__init__()
        
        # 특징 추출기
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        )
        
        # CNN 레이어
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 드롭아웃
        self.dropout = nn.Dropout(0.5)
        
        # 글로벌 평균 풀링
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 완전연결 레이어
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 특징 추출 (x: [batch_size, 1, samples])
        if x.dim() == 3:  # [batch_size, channels, samples]
            x = x.squeeze(1)  # [batch_size, samples]
        
        with torch.no_grad():
            x = self.feature_extractor(x)  # [batch_size, n_mels, time]
            x = x.unsqueeze(1)  # [batch_size, 1, n_mels, time]
        
        # CNN 레이어
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # 글로벌 평균 풀링
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 완전연결 레이어
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 