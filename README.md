# 메타버스 음성 인식 시스템

## 프로젝트 소개

이 프로젝트는 메타버스 환경에서 사용자의 음성을 인식하고 검증하는 시스템입니다. LibriSpeech 데이터셋을 사용하여 화자 식별 모델을 학습하고, 실시간으로 음성을 검증할 수 있습니다.

## 주요 기능

- LibriSpeech 데이터셋을 사용한 화자 식별 모델 학습
- 실시간 음성 검증
- 음성 데이터 전처리 및 특징 추출
- 모델 성능 평가 (정확도, EER)

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/RieLCho/DAI3105.git
cd DAI3105
```

2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. 필요한 패키지 설치

```bash
pip install torch torchaudio numpy pandas matplotlib tqdm
```

## 사용 방법

1. 전체 프로세스 실행 (데이터셋 다운로드, 학습, 테스트)

```bash
sudo python -m metaverse_voice_recognition.main --mode all
```

2. 실시간 화자 검증

```bash
python -m metaverse_voice_recognition.main --mode verify
```

3. 모델 학습만 실행

```bash
sudo python -m metaverse_voice_recognition.main --mode train --num_epochs 20
```

## 프로젝트 구조

```
metaverse_voice_recognition/
├── data/
│   └── librispeech/          # LibriSpeech 데이터셋
├── models/                   # 학습된 모델 저장
├── src/
│   └── metaverse_voice_recognition/
│       ├── main.py          # 메인 실행 파일
│       ├── model.py         # 모델 정의
│       ├── train.py         # 학습 로직
│       ├── test.py          # 테스트 로직
│       ├── dataset.py       # 데이터셋 클래스
│       └── download_librispeech.py  # 데이터셋 다운로드
└── README.md
```

## 모델 아키텍처

- 입력: 16kHz 샘플링 레이트의 오디오
- 특징 추출: Mel Spectrogram
- 모델: CNN 기반 SpeakerNet
- 출력: 화자 ID (6개 클래스)

## 성능 지표

- 학습 정확도: 68.00%
- 검증 정확도: 80.00%
- 테스트 정확도: 37.50%
- EER: 0.37%

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
