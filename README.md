# 메타버스 음성 인식 시스템

화자 인식을 위한 간단한 음성 인식 시스템입니다.

## 설치 방법

1. 가상환경 생성 및 활성화:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

2. 필요한 패키지 설치 (uv 사용):

```bash
uv pip install torch torchaudio numpy tqdm soundfile
```

## 실행 방법

### 1. main.py 사용 (권장)

모든 기능을 한 번에 실행하거나 개별적으로 실행할 수 있습니다:

```bash
# 전체 과정 실행 (데이터셋 생성 + 학습 + 테스트)
python -m metaverse_voice_recognition.main --mode all

# 데이터셋만 생성
python -m metaverse_voice_recognition.main --mode generate

# 모델만 학습
python -m metaverse_voice_recognition.main --mode train

# 모델만 테스트
python -m metaverse_voice_recognition.main --mode test
```

### 2. 개별 모듈 실행

각 기능을 개별적으로 실행할 수도 있습니다:

```bash
# 샘플 데이터셋 생성
python -m metaverse_voice_recognition.generate_sample_dataset --output_dir data/sample_dataset --num_speakers 5 --samples_per_speaker 10

# 모델 학습
python -m metaverse_voice_recognition.train --train_dir data/sample_dataset --val_dir data/sample_dataset --num_epochs 5 --batch_size 16 --save_dir models

# 모델 테스트
python -m metaverse_voice_recognition.test --model_path models/best_model.pth --test_dir data/sample_dataset
```

## 프로젝트 구조

```
metaverse/
├── data/
│   └── sample_dataset/     # 생성된 샘플 데이터셋
├── models/                 # 학습된 모델 저장
└── src/
    └── metaverse_voice_recognition/
        ├── __init__.py
        ├── dataset.py      # 데이터셋 클래스
        ├── ecapa_tdnn.py   # ECAPA-TDNN 모델
        ├── generate_sample_dataset.py  # 샘플 데이터셋 생성
        ├── train.py        # 모델 학습
        ├── test.py         # 모델 테스트
        └── main.py         # 메인 실행 파일
```

## 주의사항

- 샘플 데이터셋은 테스트 목적으로만 사용됩니다.
- 실제 사용을 위해서는 더 큰 데이터셋이 필요합니다.
- GPU가 있다면 더 빠른 학습이 가능합니다.
