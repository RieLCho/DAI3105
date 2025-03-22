import os
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, filename: str):
    """파일을 다운로드합니다."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=8192):
            size = f.write(data)
            pbar.update(size)

def extract_tar(tar_path: str, extract_path: str):
    """TAR 파일을 압축 해제합니다."""
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_path)

def prepare_librispeech():
    """LibriSpeech dev-clean 데이터셋을 다운로드하고 준비합니다."""
    # LibriSpeech dev-clean 데이터셋 다운로드 URL (약 337MB)
    base_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    
    # 다운로드 디렉토리 생성
    download_dir = Path("data/librispeech")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 다운로드
    tar_path = download_dir / "dev-clean.tar.gz"
    if not tar_path.exists():
        print("LibriSpeech dev-clean 데이터셋 다운로드 중...")
        download_file(base_url, str(tar_path))
    
    # 압축 해제
    print("압축 해제 중...")
    extract_tar(str(tar_path), str(download_dir))
    
    print("데이터셋 준비 완료!")

if __name__ == '__main__':
    prepare_librispeech() 