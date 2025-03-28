import os
import numpy as np
import torch
import torchaudio
import random
from pathlib import Path
from typing import List, Tuple
import soundfile as sf
import argparse

def generate_voice_sample(
    duration: float = 3.0,
    sample_rate: int = 16000,
    fundamental_freq: float = None,
    formants: List[Tuple[float, float]] = None
) -> torch.Tensor:
    """음성과 유사한 파형을 생성합니다."""
    t = torch.linspace(0, duration, int(duration * sample_rate))
    
    if fundamental_freq is None:
        fundamental_freq = random.uniform(85, 255)  # 일반적인 사람의 음성 범위
    
    if formants is None:
        # 랜덤한 포먼트 주파수 (일반적인 모음의 F1, F2 범위)
        formants = [
            (random.uniform(200, 800), random.uniform(0.7, 1.0)),    # F1
            (random.uniform(800, 2400), random.uniform(0.5, 0.8)),   # F2
            (random.uniform(2400, 3400), random.uniform(0.3, 0.6)),  # F3
        ]
    
    # 기본 주파수로 음성 생성
    waveform = torch.sin(2 * np.pi * fundamental_freq * t)
    
    # 포먼트 추가
    for freq, amp in formants:
        waveform += amp * torch.sin(2 * np.pi * freq * t)
    
    # 진폭 변조 (자연스러운 변화)
    envelope = 0.5 + 0.5 * torch.sin(2 * np.pi * 2 * t)
    waveform = waveform * envelope
    
    # 정규화
    waveform = waveform / torch.max(torch.abs(waveform))
    
    return waveform.unsqueeze(0)

def generate_noise_sample(
    duration: float = 3.0,
    sample_rate: int = 16000,
    noise_type: str = 'white'
) -> torch.Tensor:
    """다양한 종류의 노이즈를 생성합니다."""
    num_samples = int(duration * sample_rate)
    
    if noise_type == 'white':
        noise = torch.randn(1, num_samples) * 0.1
    elif noise_type == 'pink':
        # 주파수 도메인에서 1/f 스펙트럼을 가진 노이즈 생성
        freqs = torch.fft.fftfreq(num_samples)
        spectrum = 1 / torch.where(freqs == 0, 1e-10, torch.abs(freqs))
        phases = torch.rand(num_samples) * 2 * np.pi
        noise = torch.fft.ifft(spectrum * torch.exp(1j * phases)).real
        noise = noise.unsqueeze(0) * 0.1
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noise

def create_sample_dataset(
    output_dir: str,
    num_speakers: int = 10,
    samples_per_speaker: int = 20,
    duration: float = 3.0,
    sample_rate: int = 16000
):
    """샘플 데이터셋을 생성합니다."""
    output_dir = Path(output_dir)
    
    # 데이터셋 디렉토리 생성
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    noise_dir = output_dir / 'noise'
    
    for directory in [train_dir, test_dir, noise_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 화자별 특성 생성
    speaker_characteristics = []
    for _ in range(num_speakers):
        # 각 화자의 기본 특성
        fundamental_freq = random.uniform(85, 255)
        formants = [
            (random.uniform(200, 800), random.uniform(0.7, 1.0)),
            (random.uniform(800, 2400), random.uniform(0.5, 0.8)),
            (random.uniform(2400, 3400), random.uniform(0.3, 0.6)),
        ]
        speaker_characteristics.append((fundamental_freq, formants))
    
    # 학습 데이터 생성
    for i, (f0, formants) in enumerate(speaker_characteristics):
        speaker_dir = train_dir / f'speaker_{i:03d}'
        speaker_dir.mkdir(exist_ok=True)
        
        for j in range(samples_per_speaker):
            # 약간의 변동성 추가
            f0_var = f0 * random.uniform(0.95, 1.05)
            formants_var = [
                (f * random.uniform(0.95, 1.05), a)
                for (f, a) in formants
            ]
            
            # 음성 생성
            waveform = generate_voice_sample(
                duration=duration,
                sample_rate=sample_rate,
                fundamental_freq=f0_var,
                formants=formants_var
            )
            
            # 저장
            filename = speaker_dir / f'sample_{j:03d}.wav'
            torchaudio.save(filename, waveform, sample_rate)
    
    # 테스트 데이터 생성 (각 화자당 5개)
    for i, (f0, formants) in enumerate(speaker_characteristics):
        speaker_dir = test_dir / f'speaker_{i:03d}'
        speaker_dir.mkdir(exist_ok=True)
        
        for j in range(5):
            f0_var = f0 * random.uniform(0.95, 1.05)
            formants_var = [
                (f * random.uniform(0.95, 1.05), a)
                for (f, a) in formants
            ]
            
            waveform = generate_voice_sample(
                duration=duration,
                sample_rate=sample_rate,
                fundamental_freq=f0_var,
                formants=formants_var
            )
            
            filename = speaker_dir / f'sample_{j:03d}.wav'
            torchaudio.save(filename, waveform, sample_rate)
    
    # 노이즈 데이터 생성
    for i in range(10):
        # 백색 노이즈
        white_noise = generate_noise_sample(
            duration=duration,
            sample_rate=sample_rate,
            noise_type='white'
        )
        torchaudio.save(noise_dir / f'white_noise_{i:03d}.wav', white_noise, sample_rate)
        
        # 핑크 노이즈
        pink_noise = generate_noise_sample(
            duration=duration,
            sample_rate=sample_rate,
            noise_type='pink'
        )
        torchaudio.save(noise_dir / f'pink_noise_{i:03d}.wav', pink_noise, sample_rate)

def generate_sample_dataset(output_dir: str, num_speakers: int = 5, samples_per_speaker: int = 10):
    """샘플 데이터셋을 생성합니다."""
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 화자별 디렉토리 생성
    for speaker_id in range(num_speakers):
        speaker_dir = output_dir / f"speaker_{speaker_id:03d}"
        speaker_dir.mkdir(exist_ok=True)
        
        # 각 화자별 샘플 생성
        for sample_id in range(samples_per_speaker):
            # 1초 길이의 랜덤 음성 생성 (16kHz 샘플링 레이트)
            duration = 1.0  # 초
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # 기본 주파수 (화자마다 다르게)
            base_freq = 100 + speaker_id * 20
            
            # 음성 생성 (여러 주파수의 조합)
            signal = np.zeros_like(t)
            for freq in [base_freq, base_freq * 2, base_freq * 3]:
                signal += np.sin(2 * np.pi * freq * t)
            
            # 노이즈 추가
            noise = np.random.normal(0, 0.1, len(signal))
            signal = signal + noise
            
            # 정규화
            signal = signal / np.max(np.abs(signal))
            
            # 파일 저장
            output_file = speaker_dir / f"sample_{sample_id:03d}.wav"
            sf.write(output_file, signal, sample_rate)
    
    print(f"샘플 데이터셋 생성 완료! ({num_speakers}명의 화자, 각 {samples_per_speaker}개 샘플)")

def main():
    parser = argparse.ArgumentParser(description='샘플 데이터셋 생성')
    parser.add_argument('--output_dir', type=str, default='data/sample_dataset',
                      help='데이터셋 저장 경로')
    parser.add_argument('--num_speakers', type=int, default=5,
                      help='생성할 화자 수')
    parser.add_argument('--samples_per_speaker', type=int, default=10,
                      help='화자당 샘플 수')
    
    args = parser.parse_args()
    
    generate_sample_dataset(args.output_dir, args.num_speakers, args.samples_per_speaker)

if __name__ == '__main__':
    main() 