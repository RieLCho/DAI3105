import os
import torch
import torchaudio
import numpy as np

def generate_test_audio(duration: float = 3.0, sample_rate: int = 16000) -> torch.Tensor:
    """테스트용 가상 음성 데이터를 생성합니다."""
    # 기본 주파수 (남성 음성 ~120Hz, 여성 음성 ~210Hz)
    t = torch.linspace(0, duration, int(duration * sample_rate))
    
    # 기본 주파수와 하모닉스를 조합하여 음성과 유사한 파형 생성
    fundamental_freq = 120  # Hz
    waveform = torch.sin(2 * np.pi * fundamental_freq * t)
    
    # 하모닉스 추가
    for harmonic in [2, 3, 4]:
        waveform += 0.5 / harmonic * torch.sin(2 * np.pi * fundamental_freq * harmonic * t)
    
    # 음성의 자연스러운 변화를 시뮬레이션하기 위한 진폭 변조
    envelope = 0.5 + 0.5 * torch.sin(2 * np.pi * 2 * t)
    waveform = waveform * envelope
    
    # 정규화
    waveform = waveform / torch.max(torch.abs(waveform))
    
    return waveform.unsqueeze(0)

def generate_noise(duration: float = 3.0, sample_rate: int = 16000) -> torch.Tensor:
    """테스트용 노이즈를 생성합니다."""
    num_samples = int(duration * sample_rate)
    noise = torch.randn(1, num_samples) * 0.1
    return noise

def main():
    # 테스트 데이터 생성 설정
    sample_rate = 16000
    durations = [2.0, 2.5, 3.0, 3.5]  # 다양한 길이의 음성 생성
    
    # speaker1의 음성 생성 (낮은 주파수)
    for i, duration in enumerate(durations):
        waveform = generate_test_audio(duration, sample_rate)
        filename = os.path.join('test_data', 'speaker1', 'utterances', f'audio_{i+1}.wav')
        torchaudio.save(filename, waveform, sample_rate)
    
    # speaker2의 음성 생성 (높은 주파수 - 피치를 변경)
    for i, duration in enumerate(durations):
        waveform = generate_test_audio(duration, sample_rate)
        # 피치를 높이기 위해 시간 축을 압축 (1.5배 높은 피치)
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), 
            scale_factor=0.67, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)
        # 원래 길이에 맞게 패딩
        target_length = int(duration * sample_rate)
        if waveform.size(1) < target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_length - waveform.size(1))
            )
        else:
            waveform = waveform[:, :target_length]
        
        filename = os.path.join('test_data', 'speaker2', 'utterances', f'audio_{i+1}.wav')
        torchaudio.save(filename, waveform, sample_rate)
    
    # 노이즈 파일 생성
    for i in range(3):
        noise = generate_noise(4.0, sample_rate)
        filename = os.path.join('test_data', 'noise', f'noise_{i+1}.wav')
        torchaudio.save(filename, noise, sample_rate)

if __name__ == '__main__':
    main()
    print("테스트 데이터 생성이 완료되었습니다.") 