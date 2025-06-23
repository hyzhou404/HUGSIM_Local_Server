import zipfile
import os
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(repo_id='XDimLab/HUGSIM',revision='main',local_dir='/app/app_datas/PAMI2024/release/',local_dir_use_symlinks=False,allow_patterns=['3DRealCar/**'],repo_type='dataset')
snapshot_download(repo_id='XDimLab/HUGSIM',revision='main',local_dir='/app/app_datas/PAMI2024/release/ss',local_dir_use_symlinks=False,allow_patterns=['scenes/nuscenes/scene-0383.zip'], repo_type='dataset')

for file in Path("/app/app_datas/PAMI2024/release/ss/scenes/nuscenes").rglob("*.zip"):
    file_path = file.as_posix()
    dir_path = file.parent.as_posix()
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)
