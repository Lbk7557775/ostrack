# OSTrack 详细使用说明书

## 目录
1. [环境要求](#环境要求)
2. [环境安装](#环境安装)
3. [数据准备](#数据准备)
4. [预训练模型下载](#预训练模型下载)
5. [项目配置](#项目配置)
6. [训练模型](#训练模型)
7. [模型评估](#模型评估)
8. [可视化调试](#可视化调试)
9. [常见问题解决](#常见问题解决)
10. [性能测试](#性能测试)

## 环境要求

### 硬件要求
- **GPU**: 推荐RTX 2080Ti或更高（至少8GB显存）
- **内存**: 至少16GB RAM
- **存储**: 至少100GB可用空间（用于数据集和模型）

### 软件要求
- **操作系统**: Linux (推荐Ubuntu 18.04+) 或 Windows 10/11
- **CUDA**: 10.2 或 11.3
- **Python**: 3.8
- **Anaconda**: 推荐使用conda管理环境

## 环境安装

### 方法一：使用Conda环境文件（推荐）

```bash
# 创建并激活环境
conda env create -f ostrack_cuda113_env.yaml
conda activate ostrack
```

### 方法二：手动安装

```bash
# 创建conda环境
conda create -n ostrack python=3.8
conda activate ostrack

# 安装PyTorch (CUDA 10.2)
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

# 或者安装PyTorch (CUDA 11.3)
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch

# 运行安装脚本
bash install.sh
```

### 方法三：使用Docker

```bash
# 构建Docker镜像
docker build -t ostrack -f docker/Dockerfile .

# 运行容器
docker run --gpus all -it ostrack
```

## 数据准备

### 1. 创建数据目录结构

```bash
# 在项目根目录下创建data文件夹
mkdir -p data
cd data

# 创建各数据集的子目录
mkdir -p lasot
mkdir -p got10k/{train,val,test}
mkdir -p trackingnet
mkdir -p coco/{annotations,images}
```

### 2. 下载数据集

#### LaSOT数据集
```bash
# 下载LaSOT数据集
# 将数据解压到 data/lasot/ 目录下
# 目录结构应该是：
# data/lasot/
#   ├── airplane/
#   ├── basketball/
#   ├── bear/
#   └── ...
```

#### GOT-10K数据集
```bash
# 下载GOT-10K数据集
# 将数据解压到 data/got10k/ 目录下
# 目录结构应该是：
# data/got10k/
#   ├── train/
#   ├── val/
#   └── test/
```

#### TrackingNet数据集
```bash
# 下载TrackingNet数据集
# 将数据解压到 data/trackingnet/ 目录下
# 目录结构应该是：
# data/trackingnet/
#   ├── TRAIN_0/
#   ├── TRAIN_1/
#   ├── ...
#   ├── TRAIN_11/
#   └── TEST/
```

#### COCO数据集
```bash
# 下载COCO数据集
# 将数据解压到 data/coco/ 目录下
# 目录结构应该是：
# data/coco/
#   ├── annotations/
#   └── images/
```

## 预训练模型下载

### 1. 创建预训练模型目录
```bash
mkdir -p pretrained_models
```

### 2. 下载MAE预训练权重
```bash
# 下载MAE ViT-Base预训练权重
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -O pretrained_models/mae_pretrain_vit_base.pth
```

### 3. 下载OSTrack预训练模型（用于测试）
```bash
# 从Google Drive下载预训练模型
# 将下载的模型文件放到 output/checkpoints/train/ostrack/ 目录下
mkdir -p output/checkpoints/train/ostrack
```

## 项目配置

### 1. 设置项目路径
```bash
# 在项目根目录下运行
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

### 2. 检查配置文件
运行上述命令后，会自动生成以下配置文件：
- `lib/train/admin/local.py` - 训练相关路径配置
- `lib/test/evaluation/local.py` - 测试相关路径配置

### 3. 手动修改配置（如需要）
如果自动生成的路径不正确，可以手动编辑配置文件：

```python
# lib/train/admin/local.py
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/path/to/your/OSTrack-main'
        self.lasot_dir = '/path/to/your/data/lasot'
        self.got10k_dir = '/path/to/your/data/got10k/train'
        # ... 其他路径配置
```

## 训练模型

### 1. 单GPU训练
```bash
# 训练OSTrack-256模型
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single --use_wandb 0
```

### 2. 多GPU训练
```bash
# 使用4个GPU训练
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

### 3. 训练参数说明
- `--script`: 训练脚本名称
- `--config`: 配置文件名称（在experiments/ostrack/目录下）
- `--save_dir`: 模型保存目录
- `--mode`: 训练模式（single/multiple）
- `--nproc_per_node`: GPU数量
- `--use_wandb`: 是否使用wandb记录训练日志

### 4. 可用的配置文件
- `vitb_256_mae_ce_32x4_ep300.yaml` - OSTrack-256配置
- `vitb_384_mae_ce_32x4_ep300.yaml` - OSTrack-384配置
- `vitb_256_mae_ce_32x4_got10k_ep100.yaml` - 在GOT-10K上训练的配置

## 模型评估

### 1. 下载预训练模型
首先需要下载预训练的OSTrack模型权重。

### 2. 在LaSOT上评估
```bash
# 评估OSTrack-384在LaSOT数据集上
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4

# 分析结果
python tracking/analysis_results.py
```

### 3. 在GOT-10K上评估
```bash
# 评估OSTrack-384在GOT-10K测试集上
python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4

# 转换结果格式
python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
```

### 4. 在TrackingNet上评估
```bash
# 评估OSTrack-384在TrackingNet上
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4

# 转换结果格式
python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_ep300
```

### 5. 在单个序列上测试
```bash
# 在特定序列上测试
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --sequence 0 --threads 1 --num_gpus 1
```

## 可视化调试

### 1. 启动Visdom服务器
```bash
# 启动visdom服务器
visdom
# 服务器将在 http://localhost:8097 运行
```

### 2. 运行可视化测试
```bash
# 运行带可视化的测试
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1
```

### 3. 查看可视化结果
- 打开浏览器访问 `http://localhost:8097`
- 可以看到候选消除过程的可视化

## 性能测试

### 1. 测试模型FLOPs和参数数量
```bash
# 测试OSTrack-256
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300

# 测试OSTrack-384
python tracking/profile_model.py --script ostrack --config vitb_384_mae_ce_32x4_ep300
```

### 2. 测试推理速度
```bash
# 运行速度测试
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset lasot --threads 1 --num_gpus 1
```

## 常见问题解决

### 1. CUDA内存不足
```bash
# 减少batch size
# 编辑配置文件，将BATCH_SIZE从32改为16或8
```

### 2. 数据集路径错误
```bash
# 检查并重新设置路径
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

### 3. 预训练模型加载失败
```bash
# 确保预训练模型路径正确
# 检查pretrained_models目录下是否有mae_pretrain_vit_base.pth文件
```

### 4. 依赖包版本冲突
```bash
# 重新创建环境
conda env remove -n ostrack
conda env create -f ostrack_cuda113_env.yaml
```

### 5. Windows系统问题
- 确保使用PowerShell或CMD运行命令
- 如果遇到路径问题，使用正斜杠`/`而不是反斜杠`\`

## 快速开始示例

### 1. 完整的环境设置
```bash
# 1. 克隆项目
git clone https://github.com/botaoye/OSTrack.git
cd OSTrack

# 2. 创建环境
conda env create -f ostrack_cuda113_env.yaml
conda activate ostrack

# 3. 设置路径
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output

# 4. 下载预训练模型
mkdir -p pretrained_models
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -O pretrained_models/mae_pretrain_vit_base.pth

# 5. 准备数据集（需要手动下载）
# 将数据集放到data/目录下

# 6. 开始训练
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single --use_wandb 0
```

### 2. 快速测试
```bash
# 下载预训练模型权重到output/checkpoints/train/ostrack/
# 运行测试
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 4 --num_gpus 1
```

## 注意事项

1. **数据集下载**: 需要手动下载各个数据集，确保路径正确
2. **GPU内存**: 根据GPU显存调整batch size
3. **训练时间**: 完整训练需要较长时间，建议使用多GPU
4. **路径配置**: 确保所有路径使用绝对路径或正确的相对路径
5. **版本兼容**: 严格按照要求的Python和PyTorch版本安装

## 技术支持

如果遇到问题，可以：
1. 查看GitHub Issues
2. 检查日志文件
3. 确认环境配置是否正确
4. 验证数据集格式是否正确

---

**祝您使用愉快！** 🚀
