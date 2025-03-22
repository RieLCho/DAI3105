from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """모델 설정"""
    embedding_dim: int = 192
    channels: int = 512
    kernel_size: int = 5
    stride: int = 1
    padding: int = 2
    dilation: int = 1
    groups: int = 1

@dataclass
class TrainingConfig:
    """학습 설정"""
    train_dir: str
    val_dir: Optional[str] = None
    save_dir: str = 'models'
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    margin: float = 0.2
    scale: float = 30
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    num_workers: int = 4
    augment: bool = True
    save_interval: int = 1  # 모델 저장 간격 (에폭 단위)
    
    # 데이터 증강 관련
    noise_prob: float = 0.5
    noise_snr_range: tuple = (5, 15)
    rir_prob: float = 0.5
    speed_perturb_prob: float = 0.5
    speed_perturb_range: tuple = (0.9, 1.1)
    
    # 검증 관련
    val_interval: int = 1
    early_stopping_patience: int = 5
    
    # 기타
    seed: int = 42
    fp16: bool = True  # 혼합 정밀도 학습 사용 여부

@dataclass
class InferenceConfig:
    """추론 설정"""
    model_path: str
    sample_rate: int = 16000
    chunk_size: int = 1024
    overlap: float = 0.5  # 오버랩 비율
    threshold: float = 0.85  # 화자 식별 임계값
    
    # 특징 추출 관련
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    f_min: int = 20
    f_max: int = 7600
    n_mels: int = 80

# 기본 설정
DEFAULT_TRAIN_CONFIG = TrainingConfig(
    train_dir="data/voxceleb2/train",
    val_dir="data/voxceleb1/test",
    batch_size=64,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=0.0001,
    scheduler_step_size=20,
    scheduler_gamma=0.5,
    margin=0.2,
    scale=30,
    save_dir="models/voxceleb2",
)

DEFAULT_MODEL_CONFIG = ModelConfig(
    embedding_dim=192,
    channels=512,
    kernel_size=5,
    stride=1,
    padding=2,
    dilation=1,
    groups=1,
)

DEFAULT_INFERENCE_CONFIG = InferenceConfig(
    model_path="models/voxceleb2/best_model.pth",
    sample_rate=16000,
    chunk_size=1024,
    overlap=0.5,
    threshold=0.85,
) 