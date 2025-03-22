import os
import argparse
import subprocess
from tqdm import tqdm
import wget
import zipfile

def download_voxceleb(save_path: str, dataset: str = "vox1"):
    """VoxCeleb 데이터셋을 다운로드합니다."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # VoxCeleb1 데이터셋 URL
    vox1_urls = {
        'dev': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa',
        'test': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1_test_wav.zip',
        'meta': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv',
    }
    
    # VoxCeleb2 데이터셋 URL
    vox2_urls = {
        'dev': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa',
        'meta': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv',
    }
    
    urls = vox1_urls if dataset == "vox1" else vox2_urls
    
    print(f"다운로드 중: {dataset}")
    for name, url in urls.items():
        output_path = os.path.join(save_path, f"{dataset}_{name}.zip")
        if not os.path.exists(output_path):
            print(f"{name} 다운로드 중...")
            wget.download(url, output_path)
            print()
    
    # 압축 해제
    print("\n압축 해제 중...")
    for name in urls.keys():
        zip_path = os.path.join(save_path, f"{dataset}_{name}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_path)
            os.remove(zip_path)  # 압축 파일 삭제

def download_musan(save_path: str):
    """MUSAN 데이터셋을 다운로드합니다."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # MUSAN 데이터셋 URL
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    output_path = os.path.join(save_path, "musan.tar.gz")
    
    print("MUSAN 데이터셋 다운로드 중...")
    if not os.path.exists(output_path):
        wget.download(url, output_path)
        print()
    
    # 압축 해제
    print("\n압축 해제 중...")
    subprocess.run(['tar', '-xzf', output_path, '-C', save_path])
    os.remove(output_path)  # 압축 파일 삭제

def download_rirs(save_path: str):
    """RIRS 데이터셋을 다운로드합니다."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # RIRS 데이터셋 URL
    url = "https://www.openslr.org/resources/28/rirs_noises.zip"
    output_path = os.path.join(save_path, "rirs_noises.zip")
    
    print("RIRS 데이터셋 다운로드 중...")
    if not os.path.exists(output_path):
        wget.download(url, output_path)
        print()
    
    # 압축 해제
    print("\n압축 해제 중...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
    os.remove(output_path)  # 압축 파일 삭제

def main():
    parser = argparse.ArgumentParser(description="VoxCeleb 데이터셋 다운로더")
    parser.add_argument("--save_path", type=str, required=True, help="데이터셋을 저장할 경로")
    parser.add_argument("--dataset", type=str, default="vox1", choices=["vox1", "vox2"], help="다운로드할 데이터셋")
    parser.add_argument("--download_musan", action="store_true", help="MUSAN 데이터셋 다운로드")
    parser.add_argument("--download_rirs", action="store_true", help="RIRS 데이터셋 다운로드")
    
    args = parser.parse_args()
    
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