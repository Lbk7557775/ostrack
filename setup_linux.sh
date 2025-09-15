#!/bin/bash

echo "🚀 OSTrack Linux/Mac 快速配置脚本"
echo "================================================"

# 检查Python版本
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python版本: $PYTHON_VERSION"

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到Conda，请先安装Anaconda或Miniconda"
    exit 1
fi

echo "✅ Conda已安装"

# 创建conda环境
echo ""
echo "📦 创建conda环境..."
if conda env create -f ostrack_cuda113_env.yaml; then
    echo "✅ 环境创建成功"
else
    echo "❌ 环境创建失败，尝试手动安装..."
    echo "请运行以下命令手动创建环境:"
    echo "conda create -n ostrack python=3.8"
    echo "conda activate ostrack"
    echo "bash install.sh"
    exit 1
fi

# 激活环境并运行配置脚本
echo ""
echo "⚙️ 配置项目..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ostrack

if python3 quick_setup.py --workspace_dir . --data_dir ./data --save_dir ./output; then
    echo ""
    echo "🎉 配置完成！"
    echo ""
    echo "下一步操作:"
    echo "1. 激活环境: conda activate ostrack"
    echo "2. 下载数据集到 data/ 目录"
    echo "3. 运行训练或测试命令"
else
    echo "❌ 配置失败"
    exit 1
fi
