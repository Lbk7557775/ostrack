@echo off
chcp 65001 >nul
echo 🚀 OSTrack Windows 快速配置脚本
echo ================================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python已安装
python --version

REM 检查conda是否安装
conda --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Conda，请先安装Anaconda或Miniconda
    pause
    exit /b 1
)

echo ✅ Conda已安装

REM 创建conda环境
echo.
echo 📦 创建conda环境...
conda env create -f ostrack_cuda113_env.yaml
if errorlevel 1 (
    echo ❌ 环境创建失败，尝试手动安装...
    echo 请运行以下命令手动创建环境:
    echo conda create -n ostrack python=3.8
    echo conda activate ostrack
    echo bash install.sh
    pause
    exit /b 1
)

echo ✅ 环境创建成功

REM 激活环境并运行配置脚本
echo.
echo ⚙️ 配置项目...
call conda activate ostrack
python quick_setup.py --workspace_dir . --data_dir ./data --save_dir ./output

if errorlevel 1 (
    echo ❌ 配置失败
    pause
    exit /b 1
)

echo.
echo 🎉 配置完成！
echo.
echo 下一步操作:
echo 1. 激活环境: conda activate ostrack
echo 2. 下载数据集到 data/ 目录
echo 3. 运行训练或测试命令
echo.
echo 按任意键退出...
pause >nul
