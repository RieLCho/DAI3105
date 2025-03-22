from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """ECAPA-TDNN 모델 설정"""
    input_size: int = 80
    channels: int = 512
    emb_dim: int = 192

@dataclass
class TrainingConfig:
    """학습 설정"""
    # 데이터 관련
    train_dir: str
    val_dir: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    
    # 학습 관련
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    
    # 손실 함수 관련
    margin: float = 0.2
    scale: float = 30
    
    # 데이터 증강 관련
    augment: bool = True
    noise_prob: float = 0.5
    noise_snr_range: tuple = (5, 15)
    rir_prob: float = 0.5
    speed_perturb_prob: float = 0.5
    speed_perturb_range: tuple = (0.9, 1.1)
    
    # 모델 저장 관련
    save_dir: str = 'models'
    save_interval: int = 10
    
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
    input_size=80,
    channels=512,
    emb_dim=192,
)

DEFAULT_INFERENCE_CONFIG = InferenceConfig(
    model_path="models/voxceleb2/best_model.pth",
    sample_rate=16000,
    chunk_size=1024,
    overlap=0.5,
    threshold=0.85,
) 