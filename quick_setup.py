#!/usr/bin/env python3
"""
OSTrack 快速配置脚本
用于自动设置项目路径和检查环境
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    print(f"✅ Python版本检查通过: {sys.version}")
    return True

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("❌ CUDA不可用")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def create_directories(base_dir):
    """创建必要的目录"""
    dirs = [
        "data",
        "data/lasot",
        "data/got10k/train",
        "data/got10k/val", 
        "data/got10k/test",
        "data/trackingnet",
        "data/coco/annotations",
        "data/coco/images",
        "pretrained_models",
        "output",
        "output/checkpoints",
        "output/checkpoints/train",
        "output/checkpoints/train/ostrack"
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

def setup_paths(workspace_dir, data_dir, save_dir):
    """设置项目路径"""
    try:
        cmd = [
            sys.executable, 
            "tracking/create_default_local_file.py",
            "--workspace_dir", workspace_dir,
            "--data_dir", data_dir,
            "--save_dir", save_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 项目路径设置成功")
            return True
        else:
            print(f"❌ 路径设置失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 路径设置异常: {e}")
        return False

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        "torch", "torchvision", "opencv-python", "numpy", 
        "pandas", "tqdm", "pycocotools", "timm", "wandb"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def download_pretrained_model(pretrained_dir):
    """下载预训练模型"""
    model_path = os.path.join(pretrained_dir, "mae_pretrain_vit_base.pth")
    if os.path.exists(model_path):
        print("✅ MAE预训练模型已存在")
        return True
    
    print("正在下载MAE预训练模型...")
    try:
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
        urllib.request.urlretrieve(url, model_path)
        print("✅ MAE预训练模型下载完成")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请手动下载: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth")
        return False

def main():
    parser = argparse.ArgumentParser(description="OSTrack快速配置脚本")
    parser.add_argument("--workspace_dir", type=str, default=".", 
                       help="工作目录路径")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="数据目录路径")
    parser.add_argument("--save_dir", type=str, default="./output",
                       help="输出目录路径")
    parser.add_argument("--skip_download", action="store_true",
                       help="跳过预训练模型下载")
    
    args = parser.parse_args()
    
    print("🚀 开始配置OSTrack环境...")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 检查CUDA
    if not check_cuda():
        print("⚠️  警告: CUDA不可用，将使用CPU模式")
    
    # 检查依赖包
    print("\n📦 检查依赖包...")
    if not check_dependencies():
        return False
    
    # 创建目录
    print("\n📁 创建必要目录...")
    create_directories(args.workspace_dir)
    
    # 设置路径
    print("\n⚙️  设置项目路径...")
    workspace_abs = os.path.abspath(args.workspace_dir)
    data_abs = os.path.abspath(args.data_dir)
    save_abs = os.path.abspath(args.save_dir)
    
    if not setup_paths(workspace_abs, data_abs, save_abs):
        return False
    
    # 下载预训练模型
    if not args.skip_download:
        print("\n⬇️  下载预训练模型...")
        pretrained_dir = os.path.join(args.workspace_dir, "pretrained_models")
        download_pretrained_model(pretrained_dir)
    
    print("\n" + "=" * 50)
    print("🎉 配置完成！")
    print("\n下一步:")
    print("1. 下载数据集到 data/ 目录")
    print("2. 运行训练: python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single --use_wandb 0")
    print("3. 或运行测试: python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 4 --num_gpus 1")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
