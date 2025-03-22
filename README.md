# Metaverse Voice Recognition

메타버스 환경에서 사용할 수 있는 화자 인식 시스템입니다. ECAPA-TDNN 모델을 사용하여 화자의 목소리를 식별합니다.

## 설치 방법

1. Python 3.8 이상이 필요합니다.
2. 가상환경을 생성하고 활성화합니다:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. 필요한 패키지를 설치합니다:

```bash
pip install torch torchaudio numpy tqdm
```

## 사용 방법

### 1. 샘플 데이터셋 생성

테스트를 위한 샘플 데이터셋을 생성합니다:

```bash
python -m src.metaverse_voice_recognition.generate_sample_dataset \
    --output_dir sample_datasets \
    --num_speakers 5 \
    --samples_per_speaker 10 \
    --duration 3.0
```

### 2. 모델 학습

생성된 샘플 데이터셋으로 모델을 학습합니다:

```bash
PYTHONPATH=src python src/metaverse_voice_recognition/train.py \
    --train_dir sample_datasets/train \
    --val_dir sample_datasets/test \
    --num_epochs 10 \
    --batch_size 4 \
    --learning_rate 0.001
```

### 주요 매개변수

#### 샘플 데이터셋 생성

- `--output_dir`: 데이터셋이 저장될 디렉토리
- `--num_speakers`: 생성할 화자 수
- `--samples_per_speaker`: 화자당 생성할 샘플 수
- `--duration`: 각 음성 샘플의 길이(초)

#### 모델 학습

- `--train_dir`: 학습 데이터 디렉토리
- `--val_dir`: 검증 데이터 디렉토리
- `--num_epochs`: 학습 에폭 수
- `--batch_size`: 배치 크기
- `--learning_rate`: 학습률
- `--save_dir`: 모델이 저장될 디렉토리 (기본값: 'models')

## 프로젝트 구조

```
metaverse/
├── src/
│   └── metaverse_voice_recognition/
│       ├── train.py              # 모델 학습 스크립트
│       ├── generate_sample_dataset.py  # 샘플 데이터셋 생성
│       ├── download_voxceleb.py  # VoxCeleb 데이터셋 다운로드
│       └── ecapa_tdnn.py         # ECAPA-TDNN 모델 구현
├── sample_datasets/              # 생성된 샘플 데이터셋
│   ├── train/                    # 학습 데이터
│   └── test/                     # 테스트 데이터
└── models/                       # 학습된 모델 저장소
```

## 참고사항

- 학습 중에는 자동으로 `models` 디렉토리에 체크포인트가 저장됩니다.
- 검증 정확도가 향상되지 않으면 조기 종료됩니다.
- GPU가 있는 경우 자동으로 GPU를 사용합니다.
