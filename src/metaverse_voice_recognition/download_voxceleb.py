import os
import argparse
import subprocess
from tqdm import tqdm
import requests
import hashlib
from pathlib import Path

def download_file(url: str, output_path: str, chunk_size: int = 8192):
    """파일을 다운로드합니다."""
    response = requests.get(url, stream=True, verify=False)
    total_size = int(response.headers.get('content-length', 0))
    
    # 진행 상황 표시
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f"다운로드 중: {os.path.basename(output_path)}"
    )
    
    # 파일 다운로드
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            size = f.write(chunk)
            progress_bar.update(size)
    
    progress_bar.close()

def download_voxceleb(save_path: str, dataset: str = "vox1"):
    """VoxCeleb 데이터셋을 다운로드합니다."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # VoxCeleb1 데이터셋 URL과 MD5 해시
    vox1_files = {
        'dev': {
            'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa',
            'md5': '123c6f670c4fb5e6b3cf43fba3285d88'
        },
        'test': {
            'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1_test_wav.zip',
            'md5': '185fdc63c3c739954633d50379a3d102'
        },
        'meta': {
            'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv',
            'md5': None  # 메타데이터는 해시 검증 생략
        }
    }
    
    # VoxCeleb2 데이터셋 URL과 MD5 해시
    vox2_files = {
        'dev': {
            'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa',
            'md5': '3bf347ee8f3c4dd463ce4b546c2e147c'
        },
        'meta': {
            'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv',
            'md5': None  # 메타데이터는 해시 검증 생략
        }
    }
    
    files = vox1_files if dataset == "vox1" else vox2_files
    
    print(f"다운로드 중: {dataset}")
    for name, info in files.items():
        output_path = os.path.join(save_path, f"{dataset}_{name}.zip")
        
        # 이미 다운로드된 파일이 있고 해시가 일치하면 건너뛰기
        if os.path.exists(output_path) and info['md5']:
            with open(output_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash == info['md5']:
                print(f"{name} 파일이 이미 존재합니다. 건너뜁니다.")
                continue
        
        # 파일 다운로드
        print(f"{name} 다운로드 중...")
        try:
            download_file(info['url'], output_path)
        except Exception as e:
            print(f"다운로드 실패: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue
        
        # 해시 검증
        if info['md5']:
            with open(output_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash != info['md5']:
                print(f"해시 불일치: {name}")
                os.remove(output_path)
                continue
        
        # 압축 해제
        if output_path.endswith('.zip'):
            print(f"{name} 압축 해제 중...")
            subprocess.run(['unzip', '-q', output_path, '-d', save_path])
            os.remove(output_path)

def download_musan(save_path: str):
    """MUSAN 데이터셋을 다운로드합니다."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # MUSAN 데이터셋 URL
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    output_path = os.path.join(save_path, "musan.tar.gz")
    
    print("MUSAN 데이터셋 다운로드 중...")
    try:
        download_file(url, output_path)
        
        # 압축 해제
        print("\n압축 해제 중...")
        subprocess.run(['tar', '-xzf', output_path, '-C', save_path])
        os.remove(output_path)
    except Exception as e:
        print(f"다운로드 실패: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)

def download_rirs(save_path: str):
    """RIRS 데이터셋을 다운로드합니다."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # RIRS 데이터셋 URL
    url = "https://www.openslr.org/resources/28/rirs_noises.zip"
    output_path = os.path.join(save_path, "rirs_noises.zip")
    
    print("RIRS 데이터셋 다운로드 중...")
    try:
        download_file(url, output_path)
        
        # 압축 해제
        print("\n압축 해제 중...")
        subprocess.run(['unzip', '-q', output_path, '-d', save_path])
        os.remove(output_path)
    except Exception as e:
        print(f"다운로드 실패: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)

def main():
    parser = argparse.ArgumentParser(description="VoxCeleb 데이터셋 다운로더")
    parser.add_argument("--save_path", type=str, required=True, help="데이터셋을 저장할 경로")
    parser.add_argument("--dataset", type=str, default="vox1", choices=["vox1", "vox2"], help="다운로드할 데이터셋")
    parser.add_argument("--download_musan", action="store_true", help="MUSAN 데이터셋 다운로드")
    parser.add_argument("--download_rirs", action="store_true", help="RIRS 데이터셋 다운로드")
    
    args = parser.parse_args()
    
    # requests의 경고 메시지 무시
    import warnings
    warnings.filterwarnings("ignore")
    
    # VoxCeleb 다운로드
    download_voxceleb(args.save_path, args.dataset)
    
    # MUSAN 다운로드 (선택사항)
    if args.download_musan:
        musan_path = os.path.join(args.save_path, "musan")
        download_musan(musan_path)
    
    # RIRS 다운로드 (선택사항)
    if args.download_rirs:
        rirs_path = os.path.join(args.save_path, "rirs_noises")
        download_rirs(rirs_path)
    
    print("다운로드가 완료되었습니다!")

if __name__ == "__main__":
    main() 