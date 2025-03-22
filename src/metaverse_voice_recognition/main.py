import argparse
import sounddevice as sd
import numpy as np
import wave
import os
from pathlib import Path
from .generate_sample_dataset import generate_sample_dataset
from .train import train_model
from .test import evaluate_model
from .config import TrainingConfig
from .download_librispeech import prepare_librispeech
from .model import SpeakerNet
import torch
import torchaudio

def record_audio(duration=5, sample_rate=16000, output_path="temp_recording.wav"):
    """실시간 음성 녹음"""
    print(f"\n{duration}초 동안 음성을 녹음합니다...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    print("녹음 완료!")
    
    # WAV 파일로 저장
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())
    
    return output_path

def verify_speaker(model_path, audio_path, device='cpu'):
    """화자 검증 수행"""
    # 모델 로드
    model = SpeakerNet(num_classes=6).to(device)  # 실제 화자 수로 변경
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 오디오 로드 및 전처리
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # 모델 추론
    with torch.no_grad():
        output = model(waveform)
        _, predicted = torch.max(output, 1)
        
    return predicted.item()

def main():
    parser = argparse.ArgumentParser(description='메타버스 음성 인식 시스템')
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['download', 'generate', 'train', 'test', 'all', 'verify'],
                      help='실행할 모드: download(데이터셋 다운로드), generate(샘플 데이터셋 생성), train(모델 학습), test(모델 테스트), all(전체 실행), verify(실시간 검증)')
    parser.add_argument('--output_dir', type=str, default='data/librispeech',
                      help='데이터셋 저장 경로')
    parser.add_argument('--num_speakers', type=int, default=5,
                      help='생성할 화자 수')
    parser.add_argument('--samples_per_speaker', type=int, default=10,
                      help='화자당 샘플 수')
    parser.add_argument('--num_epochs', type=int, default=5,
                      help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='배치 크기')
    parser.add_argument('--save_dir', type=str, default='models',
                      help='모델 저장 경로')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                      help='테스트할 모델 경로')
    parser.add_argument('--record_duration', type=int, default=5,
                      help='녹음 시간 (초)')
    
    args = parser.parse_args()
    
    # 데이터셋 다운로드
    if args.mode in ['download', 'all']:
        print("\n1. LibriSpeech 데이터셋 다운로드 중...")
        prepare_librispeech()
    
    # 샘플 데이터셋 생성 (테스트용)
    if args.mode in ['generate', 'all']:
        print("\n2. 샘플 데이터셋 생성 중...")
        generate_sample_dataset(
            args.output_dir,
            args.num_speakers,
            args.samples_per_speaker
        )
    
    # 모델 학습
    if args.mode in ['train', 'all']:
        print("\n3. 모델 학습 중...")
        config = TrainingConfig(
            train_dir=args.output_dir,
            val_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_dir=args.save_dir
        )
        train_model(config)
    
    # 모델 테스트
    if args.mode in ['test', 'all']:
        print("\n4. 모델 테스트 중...")
        evaluate_model(
            model_path=args.model_path,
            test_dir=args.output_dir
        )
    
    # 실시간 화자 검증
    if args.mode == 'verify':
        print("\n실시간 화자 검증을 시작합니다...")
        while True:
            try:
                # 음성 녹음
                audio_path = record_audio(duration=args.record_duration)
                
                # 화자 검증
                speaker_id = verify_speaker(args.model_path, audio_path)
                print(f"예측된 화자 ID: {speaker_id}")
                
                # 임시 파일 삭제
                os.remove(audio_path)
                
                # 계속할지 확인
                response = input("\n계속 검증하시겠습니까? (y/n): ")
                if response.lower() != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n검증을 종료합니다.")
                break

if __name__ == '__main__':
    main() 