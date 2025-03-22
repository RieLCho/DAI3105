import numpy as np
import time
import torch
from metaverse_voice_recognition.config import InferenceConfig
from metaverse_voice_recognition.ecapa_tdnn import ECAPA_TDNN, VoiceEncoder
from metaverse_voice_recognition.streaming import VoiceDetector

def generate_test_audio(duration: float = 3.0, sample_rate: int = 16000, fundamental_freq: float = 120) -> np.ndarray:
    """테스트용 가상 음성 데이터를 생성합니다."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # 기본 주파수와 하모닉스를 조합하여 음성과 유사한 파형 생성
    waveform = np.sin(2 * np.pi * fundamental_freq * t)
    
    for harmonic in [2, 3, 4]:
        waveform += 0.5 / harmonic * np.sin(2 * np.pi * fundamental_freq * harmonic * t)
    
    # 음성의 자연스러운 변화를 시뮬레이션하기 위한 진폭 변조
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    waveform = waveform * envelope
    
    # 정규화
    waveform = waveform / np.max(np.abs(waveform))
    
    return waveform

def test_voice_detector():
    """화자 검출기를 테스트합니다."""
    print("화자 검출기 테스트 시작...")
    
    # 설정 및 모델 초기화
    config = InferenceConfig(model_path="models/test_model.pth")
    model = ECAPA_TDNN()
    voice_encoder = VoiceEncoder(model_path=None)  # 테스트용으로 랜덤 가중치 사용
    detector = VoiceDetector(config, voice_encoder)
    
    # 테스트 오디오 생성
    print("테스트 오디오 생성 중...")
    reference_audio = generate_test_audio(duration=3.0)
    test_audio_same = generate_test_audio(duration=3.0)  # 같은 특성의 음성
    test_audio_diff = generate_test_audio(duration=3.0, fundamental_freq=240)  # 다른 특성의 음성
    
    # 검출기 시작
    print("검출기 시작...")
    detector.start()
    
    try:
        # 기준 음성 등록
        print("기준 음성 처리 중...")
        chunk_size = 1024
        for i in range(0, len(reference_audio), chunk_size):
            chunk = reference_audio[i:i + chunk_size]
            detector.process_audio(chunk)
            time.sleep(0.01)  # 실제 스트리밍 시뮬레이션
        
        # 같은 특성의 음성 테스트
        print("\n같은 특성의 음성 테스트 중...")
        detections_same = []
        for i in range(0, len(test_audio_same), chunk_size):
            chunk = test_audio_same[i:i + chunk_size]
            result = detector.process_audio(chunk)
            detections_same.append(result)
            time.sleep(0.01)
        
        # 다른 특성의 음성 테스트
        print("\n다른 특성의 음성 테스트 중...")
        detections_diff = []
        for i in range(0, len(test_audio_diff), chunk_size):
            chunk = test_audio_diff[i:i + chunk_size]
            result = detector.process_audio(chunk)
            detections_diff.append(result)
            time.sleep(0.01)
        
        # 결과 분석
        print("\n테스트 결과:")
        if detections_same:
            print(f"같은 화자 탐지율: {sum(detections_same)/len(detections_same)*100:.1f}%")
        if detections_diff:
            print(f"다른 화자 오탐지율: {sum(detections_diff)/len(detections_diff)*100:.1f}%")
        
    finally:
        detector.stop()
        print("\n테스트 완료")

if __name__ == "__main__":
    test_voice_detector() 