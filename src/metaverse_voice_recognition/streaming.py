import torch
import numpy as np
import threading
import queue
import time
from typing import Optional, List, Tuple
from collections import deque

from .config import InferenceConfig
from .ecapa_tdnn import VoiceEncoder

class AudioBuffer:
    """오디오 버퍼 클래스"""
    
    def __init__(self, max_size: int = 48000):  # 3초 @ 16kHz
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_samples(self, samples: np.ndarray):
        """샘플을 버퍼에 추가합니다."""
        with self.lock:
            for sample in samples:
                self.buffer.append(sample)
    
    def get_samples(self, num_samples: Optional[int] = None) -> np.ndarray:
        """버퍼에서 샘플을 가져옵니다."""
        with self.lock:
            if num_samples is None:
                return np.array(self.buffer)
            return np.array(list(self.buffer)[-num_samples:])
    
    def clear(self):
        """버퍼를 비웁니다."""
        with self.lock:
            self.buffer.clear()

class StreamProcessor:
    """실시간 음성 스트림 처리기"""
    
    def __init__(self, config: InferenceConfig, voice_encoder: VoiceEncoder):
        self.config = config
        self.voice_encoder = voice_encoder
        
        # 오디오 버퍼 초기화
        self.buffer = AudioBuffer(
            max_size=int(config.sample_rate * 5)  # 최대 5초
        )
        
        # 처리 스레드 설정
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None
        
        # 임베딩 캐시
        self.cached_embeddings: List[torch.Tensor] = []
        self.cached_timestamps: List[float] = []
        self.max_cache_size = 100
    
    def start(self):
        """처리를 시작합니다."""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.start()
    
    def stop(self):
        """처리를 중지합니다."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def add_audio(self, audio_chunk: np.ndarray):
        """오디오 청크를 추가합니다."""
        self.buffer.add_samples(audio_chunk)
        self.processing_queue.put(time.time())
    
    def _process_stream(self):
        """스트림을 처리합니다."""
        window_size = int(self.config.sample_rate * 3)  # 3초 윈도우
        hop_size = int(window_size * (1 - self.config.overlap))  # 오버랩 적용
        
        while self.is_running:
            try:
                timestamp = self.processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # 현재 버퍼에서 오디오 가져오기
            audio_data = self.buffer.get_samples()
            
            # 윈도우가 충분히 차있는지 확인
            if len(audio_data) < window_size:
                continue
            
            # 윈도우 처리
            start_idx = max(0, len(audio_data) - window_size)
            window = audio_data[start_idx:]
            
            # 임베딩 생성
            audio_tensor = torch.FloatTensor(window).unsqueeze(0)
            embedding = self.voice_encoder.encode_voice(audio_tensor)
            
            # 캐시 업데이트
            self._update_cache(embedding, timestamp)
            
            # 유사도 계산 및 결과 전송
            similarity = self._compute_max_similarity(embedding)
            self.results_queue.put((timestamp, similarity))
    
    def _update_cache(self, embedding: torch.Tensor, timestamp: float):
        """임베딩 캐시를 업데이트합니다."""
        self.cached_embeddings.append(embedding)
        self.cached_timestamps.append(timestamp)
        
        # 캐시 크기 제한
        if len(self.cached_embeddings) > self.max_cache_size:
            self.cached_embeddings.pop(0)
            self.cached_timestamps.pop(0)
    
    def _compute_max_similarity(self, embedding: torch.Tensor) -> float:
        """캐시된 임베딩들과의 최대 유사도를 계산합니다."""
        if not self.cached_embeddings:
            return 0.0
        
        similarities = [
            self.voice_encoder.compute_similarity(embedding, cached_emb)
            for cached_emb in self.cached_embeddings
        ]
        return max(similarities)
    
    def get_results(self) -> List[Tuple[float, float]]:
        """처리 결과를 가져옵니다."""
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        return results

class VoiceDetector:
    """화자 검출기"""
    
    def __init__(self, config: InferenceConfig, voice_encoder: VoiceEncoder):
        self.config = config
        self.stream_processor = StreamProcessor(config, voice_encoder)
        self.detection_threshold = config.threshold
        self.detection_window = 5  # 5개의 연속된 프레임에서 탐지
        self.recent_detections = deque(maxlen=self.detection_window)
    
    def start(self):
        """검출을 시작합니다."""
        self.stream_processor.start()
    
    def stop(self):
        """검출을 중지합니다."""
        self.stream_processor.stop()
    
    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """오디오를 처리하고 화자 검출 여부를 반환합니다."""
        self.stream_processor.add_audio(audio_chunk)
        
        # 결과 처리
        results = self.stream_processor.get_results()
        if not results:
            return False
        
        # 최근 탐지 결과 업데이트
        for _, similarity in results:
            self.recent_detections.append(similarity > self.detection_threshold)
        
        # 연속된 프레임에서의 탐지 여부 확인
        if len(self.recent_detections) < self.detection_window:
            return False
        
        return sum(self.recent_detections) >= self.detection_window * 0.6  # 60% 이상에서 탐지되면 True 