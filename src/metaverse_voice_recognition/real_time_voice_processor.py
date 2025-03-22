import pyaudio
import wave
import numpy as np
import torch
import threading
import queue
import time
from typing import Optional

class RealTimeVoiceProcessor:
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 record_seconds: int = 5):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.record_seconds = record_seconds
        self.format = pyaudio.paFloat32
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def start_recording(self):
        """실시간 음성 녹음을 시작합니다."""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.is_recording = True
        self.stream.start_stream()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 스트림에서 데이터를 받아 큐에 저장합니다."""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self):
        """녹음을 중지합니다."""
        if self.stream:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """큐에서 오디오 청크를 가져옵니다."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
            
    def __del__(self):
        """정리 작업을 수행합니다."""
        self.stop_recording()
        self.audio.terminate()

    def save_audio(self, filename: str, audio_data: np.ndarray):
        """오디오 데이터를 WAV 파일로 저장합니다."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes()) 