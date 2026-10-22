# CTSC-FC-SiamKPConv

> 完整源码包：[CTSC-FC-SiamKPConv-source.zip](CTSC-FC-SiamKPConv-source.zip)。下载并解压后，在解压得到的项目根目录安装依赖、配置数据并运行训练。默认输入为XYZ＋CTSC，融合缩放因子0.25，FEAT=13。

跨时相先验特征增强的三维点云变化检测方法。

本项目提供上述论文的相关实现，基于FC-SiamKPConv开展双时相点云的多类别变化检测。方法在保留Siamese KPConv主体架构的基础上，完善输入端的先验特征计算模块（FCM）：构建跨时相先验特征CTSC，并通过独立标准化与加权拼接，将先验信息与XYZ坐标融合后输入网络。

## 方法概述

单时相描述符能够补充点云的局部结构信息，但难以直接表达两期点云之间的变化。本文从跨时相邻域关系出发，构造CTSC，并统一不同输入特征的数值尺度。主要包括：

- **跨时相先验特征CTSC**：联合描述几何差异、高度变化及局部对应关系，形成10维特征。
- **独立标准化与加权融合**：分别对XYZ与所选先验特征进行Z-Score标准化，对先验特征施加缩放因子后拼接。
- **多类别变化分割**：利用Siamese KPConv提取双时相点云特征，预测城市场景中的变化类别。

默认输入流程：

```text
XYZ (3-D) + 0.25 × CTSC (10-D) → 13-D fused input → Siamese KPConv
```

### CTSC特征

CTSC在另一时相中搜索局部近邻，无需两期点云具有严格的一一对应点。邻域点数取 **k=8**，由以下三组特征构成：

| 特征组 | 维度 | 描述内容 |
| --- | ---: | --- |
| 几何差异 | 4 | 三维最近距离、邻域距离均值与标准差、水平最近距离 |
| 高度变化 | 3 | 高差中位数、绝对高差中位数、高差四分位距 |
| 局部对应 | 3 | 局部尺度比、双向最近邻一致性、距离置信度 |

各组特征分别标准化，截断至 `[-5, 5]`，再乘以内部缩放系数 `0.5`。该系数与融合阶段的缩放因子作用于不同阶段。

### 特征融合

对于选定的先验特征，融合输入可写为：

```text
X = concat(ZScore(XYZ), λf × ZScore(Ff))
```

其中，`Ff`为所选先验特征，`λf`为对应的融合缩放因子。论文分别研究XYZ、CTSC、XYZ+LBP、XYZ+FPFH、XYZ+SHOT和XYZ+CTSC六种输入设置。

## 论文实验结果

论文在 **Urb3DCD-V2** 数据集上进行七类别变化检测实验。下表摘自论文表1，数值单位为%。各组模型根据验证集的变化类别平均交并比选择检查点，再在测试集上评估。

| 输入特征 | mAcc | mIoU | mIoUch |
| --- | ---: | ---: | ---: |
| XYZ | 90.72 | 81.95 | 79.65 |
| XYZ+LBP | 92.83 | 83.58 | 81.48 |
| XYZ+FPFH | 92.23 | 83.94 | 81.92 |
| XYZ+SHOT | 91.82 | 83.41 | 81.34 |
| CTSC | 92.99 | 87.63 | 85.95 |
| **XYZ+CTSC** | **95.12** | **89.41** | **87.97** |

XYZ+CTSC的mIoUch较XYZ基线提高 **8.32个百分点**。`mIoU`表示全部七个类别的IoU均值，`mIoUch`表示除未变化类别外六个变化类别的IoU均值。

论文权重分析中，XYZ+LBP、XYZ+FPFH、XYZ+SHOT和XYZ+CTSC在已测试取值中的最优融合缩放因子分别为 `0.7`、`0.5`、`0.5` 和 `0.25`。这些数值为论文报告结果；运行其他权重的实验时应分别记录配置与结果。

## 环境准备

项目使用PyTorch、PyTorch Geometric、Torch-Points3D相关组件和Hydra配置系统。依赖版本参考 [requirements.txt](requirements.txt)，其中包含：

| 组件 | 依赖文件中的版本 |
| --- | --- |
| PyTorch / torchvision | 1.8.1 / 0.9.1 |
| torch-geometric | 1.7.2 |
| torch-points-kernels | 0.7.0 |
| Hydra / OmegaConf | 1.0.7 / 2.0.6 |

在与上述依赖兼容的Python与CUDA环境中安装：

```bash
pip install -r requirements.txt
```

`torch-scatter`、`torch-sparse`、`torch-cluster`和`torch-points-kernels`含编译扩展，应与PyTorch和CUDA版本匹配。已有可运行环境时可直接使用该环境。

## 项目结构

```text
CTSC-FC-SiamKPConv/
├── trainSiamKPConv.py                         # 训练入口
├── eval_SiamKPConv.py                         # 评估入口
├── conf/
│   ├── configSiamKPConv.yaml                  # 训练主配置
│   ├── evalSiamKPConv.yaml                    # 评估配置
│   ├── data/change_detection/Urb3DSimulPair.yaml
│   ├── models/change_detection/siamesekpconv.yaml
│   ├── training/defaultUrb3D.yaml
│   └── hydra/output/custom.yaml               # 训练输出目录
├── torch_points3d/
│   ├── core/data_transform/features.py        # CTSC与其他描述符、加权融合
│   ├── datasets/change_detection/            # 数据加载与采样
│   ├── models/change_detection/siamesekpconv.py
│   ├── modules/KPConv/                       # KPConv模块
│   └── metrics/                              # 指标计算与结果导出
├── scripts/
│   ├── train.sh
│   └── eval.sh
├── requirements.txt
└── LICENSE.md
```

## 数据准备

论文使用Urb3DCD-V2合成航空LiDAR点云变化检测数据集，包含未变化、新建筑、建筑物拆除、新植被、植被生长、缺失植被和移动对象七类。数据背景与来源见 [Urb3DCD数据集论文](https://www.mdpi.com/2072-4292/13/13/2629) 及 [Siamese KPConv项目](https://github.com/IdeGelis/torch-points3d-SiameseKPConv)。

将双时相点云按以下结构组织：

```text
Urb3DSimu/1-Lidar05/
├── Train/
│   └── <scene>/
│       ├── pointCloud0.ply
│       └── pointCloud1.ply
├── Val/
│   └── <scene>/
│       ├── pointCloud0.ply
│       └── pointCloud1.ply
└── Test/
    └── <scene>/
        ├── pointCloud0.ply
        └── pointCloud1.ply
```

当前加载器使用PLY中的 `params` 元素，读取 `x`、`y`、`z` 坐标及 `label_ch` 变化标签。`pointCloud0.ply`与`pointCloud1.ply`分别对应两个时相。

在 `conf/data/change_detection/Urb3DSimulPair.yaml` 中设置：

```yaml
data:
  dataset_root: /path/to/Urb3DSimu/1-Lidar05
  dataroot: /path/to/CTSC-FC-SiamKPConv/data
  preprocessed_dir: /path/to/CTSC-FC-SiamKPConv/data/pos_ctsc_mufeats0.25
  load_preprocessed: True
  load_trees: True
```

训练集、验证集与测试集路径由 `dataset_root` 自动展开。已有对应缓存时复用；缓存缺失时自动预处理并构建KD树。修改数据、特征组合、融合权重或采样设置后，应使用新的 `preprocessed_dir`。

## 训练

在项目根目录运行：

```bash
python trainSiamKPConv.py
```

也可通过脚本指定数据目录：

```bash
bash scripts/train.sh /path/to/Urb3DSimu/1-Lidar05
```

默认实验为 **XYZ+CTSC**：XYZ为3维，CTSC为10维，`FEAT=13`，CTSC融合缩放因子为 `0.25`。CTSC包含完整的几何差异、高度变化及局部对应信息三组特征。

### 输入配置

切换输入特征时，需要同时修改数据配置和模型配置，并使用对应缓存。下表列出六种输入设置；SHOT按当前实现保留前32维：

| 输入 | `data.feat_names` | `data.mufeats_comps` | `FEAT` |
| --- | --- | --- | ---: |
| XYZ | `[pos, mufeats]` | `pos` | 3 |
| XYZ+LBP | `[pos, lbp, mufeats]` | `pos+lbp` | 11 |
| XYZ+FPFH | `[pos, fpfh, mufeats]` | `pos+fpfh` | 48 |
| XYZ+SHOT | `[pos, shot, mufeats]` | `pos+shot` | 35 |
| CTSC | `[ctsc, mufeats]` | `ctsc` | 10 |
| XYZ+CTSC | `[pos, ctsc, mufeats]` | `pos+ctsc` | 13 |

`FEAT`位于 `conf/models/change_detection/siamesekpconv.yaml` 的 `models.SiameseKPConv.define_constants` 中，表示输入点属性的维数，不包含模型自动添加的常数1通道。点的空间坐标始终参与KPConv的邻域与核点计算。

表中CTSC单独输入的设置同样经过 `MuFeatsCompute` 标准化与加权，不拼接XYZ属性，输出仍为10维。

融合缩放因子在 `torch_points3d/core/data_transform/features.py` 的 `MuFeatsCompute.weights` 中设置。切换实验时同步更新 `conf/configSiamKPConv.yaml` 中的 `job_name` 和特征缓存目录。

### 训练参数

论文第4.1节采用以下设置，对应配置位于 `conf/training/defaultUrb3D.yaml` 和 `conf/lr_scheduler/exponential.yaml`：

| 参数 | 设置 |
| --- | --- |
| `epochs` | 200 |
| Batch size | 10 |
| 优化器 | SGD |
| 初始学习率 | 0.01 |
| Momentum | 0.98 |
| Weight decay | 0.001 |
| 学习率指数衰减系数 | 0.9885 |
| 梯度裁剪阈值 | 100 |
| BN初始动量 | 0.02 |
| BN动量衰减 | step_decay，系数0.9，间隔1000，下限0.01 |

当前配置另保留 `training.optim.early_stop=100`，实际训练结束轮次受早停及训练循环控制。

### 输出目录

在 `conf/hydra/output/custom.yaml` 中设置输出根目录。训练日志、检查点和运行配置按以下结构保存：

```text
outputs/
└── <job_name>/
    └── <job_name>-SiameseKPConv-<timestamp>/
        ├── .hydra/
        ├── SiameseKPConv.pt
        └── ...
```

公开版本以启动时的项目根目录为路径基准。数据默认放置在 `data/Urb3DSimu/1-Lidar05`，也可通过上面的脚本传入数据路径；预处理结果保存在 `data/pos_ctsc_mufeats0.25`，训练结果保存在 `outputs`。数据集、缓存、模型权重和训练输出不随代码提供。

## 评估

指定训练运行目录，加载验证集上 `miou_ch` 最优的检查点进行评估：

```bash
python eval_SiamKPConv.py \
  checkpoint_dir=/path/to/training_run \
  weight_name=miou_ch
```

或使用：

```bash
bash scripts/eval.sh /path/to/training_run
```

`training_run`目录应包含 `SiameseKPConv.pt`。评估从检查点恢复模型及数据配置，默认开启完整点云与全分辨率评估，并导出预测点云。更换待评估实验时应显式指定对应运行目录。

结果保存在：

```text
<training_run>/eval/<date_time>/
```

主要指标包括mAcc、mIoU、mIoUch及各类别IoU。

## 致谢

本项目基于 [Torch-Points3D](https://github.com/torch-points3d/torch-points3d) 和 [SiamKPConvVariants](https://github.com/IdeGelis/torch-points3d-SiamKPConvVariants) 开发，感谢原作者公开代码。

项目许可证见 [LICENSE.md](LICENSE.md)，第三方代码来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
