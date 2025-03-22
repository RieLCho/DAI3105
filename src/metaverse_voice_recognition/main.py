import argparse
from pathlib import Path
from .generate_sample_dataset import generate_sample_dataset
from .train import train_model
from .test import evaluate_model
from .config import TrainingConfig

def main():
    parser = argparse.ArgumentParser(description='메타버스 음성 인식 시스템')
    parser.add_argument('--mode', type=str, required=True, choices=['generate', 'train', 'test', 'all'],
                      help='실행할 모드: generate(데이터셋 생성), train(모델 학습), test(모델 테스트), all(전체 실행)')
    parser.add_argument('--output_dir', type=str, default='data/sample_dataset',
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
    
    args = parser.parse_args()
    
    # 데이터셋 생성
    if args.mode in ['generate', 'all']:
        print("\n1. 샘플 데이터셋 생성 중...")
        generate_sample_dataset(
            args.output_dir,
            args.num_speakers,
            args.samples_per_speaker
        )
    
    # 모델 학습
    if args.mode in ['train', 'all']:
        print("\n2. 모델 학습 중...")
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
        print("\n3. 모델 테스트 중...")
        evaluate_model(
            model_path=args.model_path,
            test_dir=args.output_dir
        )

if __name__ == '__main__':
    main() 