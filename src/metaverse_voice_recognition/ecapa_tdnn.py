import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Res2Block(nn.Module):
    def __init__(self, channels, kernel_size=3, scale=8):
        super(Res2Block, self).__init__()
        self.scale = scale
        self.width = channels // scale
        self.nums = scale

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, padding=kernel_size//2)
            for _ in range(self.nums - 1)
        ])
        self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        out = spx[0]
        
        for i in range(self.nums - 1):
            if i == 0:
                sp = spx[i + 1]
            else:
                sp = sp + spx[i + 1]
            sp = self.convs[i](sp)
            out = torch.cat((out, sp), 1)
        
        return out

class ECAPA_TDNN(nn.Module):
    def __init__(self, input_size=80, channels=512, emb_dim=192):
        super(ECAPA_TDNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.layer1 = nn.Sequential(
            Res2Block(channels),
            SEModule(channels),
            nn.BatchNorm1d(channels)
        )
        
        self.layer2 = nn.Sequential(
            Res2Block(channels),
            SEModule(channels),
            nn.BatchNorm1d(channels)
        )
        
        self.layer3 = nn.Sequential(
            Res2Block(channels),
            SEModule(channels),
            nn.BatchNorm1d(channels)
        )
        
        self.embedding = nn.Linear(3*channels, emb_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # Global statistics pooling
        global_x = torch.cat((x1,x2,x3), dim=1)
        w = F.adaptive_avg_pool1d(global_x, 1)
        
        # Embedding
        emb = self.embedding(w.view(w.size(0), -1))
        
        return F.normalize(emb, p=2, dim=1)

class VoiceEncoder:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ECAPA_TDNN().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # MFCC 특징 추출기 초기화
        self.feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=80,
            melkwargs={'n_fft': 512, 'hop_length': 160}
        )
        
    @torch.no_grad()
    def encode_voice(self, audio_data: torch.Tensor) -> torch.Tensor:
        """음성 데이터를 임베딩 벡터로 변환합니다."""
        # MFCC 특징 추출
        features = self.feature_extractor(audio_data)
        features = features.unsqueeze(0).to(self.device)
        
        # 임베딩 생성
        embedding = self.model(features)
        return embedding
        
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """두 임베딩 벡터 간의 코사인 유사도를 계산합니다."""
        return F.cosine_similarity(emb1, emb2).item() 