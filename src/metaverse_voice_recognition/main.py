import torch
import numpy as np
import time
from real_time_voice_processor import RealTimeVoiceProcessor
from ecapa_tdnn import VoiceEncoder

def main():
    # 음성 처리기와 인코더 초기화
    voice_processor = RealTimeVoiceProcessor()
    voice_encoder = VoiceEncoder()
    
    print("=== 메타버스 음성 인식 시스템 ===")
    print("1. 등록된 음성 녹음 (5초)")
    print("2. 실시간 음성 비교")
    print("q. 종료")
    
    registered_embedding = None
    
    while True:
        choice = input("\n선택하세요: ")
        
        if choice == 'q':
            break
            
        elif choice == '1':
            print("5초 동안 음성을 녹음합니다...")
            voice_processor.start_recording()
            
            # 5초 동안 음성 데이터 수집
            audio_chunks = []
            start_time = time.time()
            while time.time() - start_time < 5:
                chunk = voice_processor.get_audio_chunk()
                if chunk is not None:
                    audio_chunks.append(chunk)
                    
            voice_processor.stop_recording()
            
            # 수집된 음성 데이터를 텐서로 변환
            audio_data = np.concatenate(audio_chunks)
            audio_tensor = torch.FloatTensor(audio_data)
            
            # 음성 임베딩 생성
            registered_embedding = voice_encoder.encode_voice(audio_tensor)
            print("음성이 성공적으로 등록되었습니다!")
            
        elif choice == '2':
            if registered_embedding is None:
                print("먼저 음성을 등록해주세요!")
                continue
                
            print("실시간 음성 비교를 시작합니다. 종료하려면 Ctrl+C를 누르세요.")
            voice_processor.start_recording()
            
            try:
                while True:
                    # 3초마다 음성 비교
                    audio_chunks = []
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        chunk = voice_processor.get_audio_chunk()
                        if chunk is not None:
                            audio_chunks.append(chunk)
                    
                    if audio_chunks:
                        # 수집된 음성 데이터를 텐서로 변환
                        audio_data = np.concatenate(audio_chunks)
                        audio_tensor = torch.FloatTensor(audio_data)
                        
                        # 실시간 음성 임베딩 생성 및 유사도 계산
                        current_embedding = voice_encoder.encode_voice(audio_tensor)
                        similarity = voice_encoder.compute_similarity(
                            registered_embedding, current_embedding
                        )
                        
                        print(f"유사도: {similarity:.4f}")
                        
                        # 유사도가 높으면 경고
                        if similarity > 0.85:
                            print("⚠️ 등록된 화자와 매우 유사한 음성이 감지되었습니다!")
                            
            except KeyboardInterrupt:
                voice_processor.stop_recording()
                print("\n실시간 비교를 종료합니다.")
                
if __name__ == "__main__":
    main() 