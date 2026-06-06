# 番茄病害识别系统 (Tomato Disease Recognition System)

基于 PyTorch 的番茄叶片病害图像分类系统，作为"人工智能基础(A)"课程大作业实现。
支持多种卷积神经网络架构、两阶段迁移学习策略、Grad-CAM 可视化解释，并附带 Gradio 交互式前端。

---

## 项目概述

本项目实现了一个完整的番茄叶片病害图像分类流程，涵盖：

- **数据增强**：随机水平翻转、随机旋转、ImageNet 标准化
- **两阶段迁移学习**：先冻结骨干网络微调分类头，再解冻深层联合微调
- **多架构支持**：ResNet-18、ResNet-50、MobileNet-V3 Small
- **实验对比**：从零训练 vs 迁移学习、不同冻结策略、不同学习率、不同骨干网络
- **模型可解释性**：Grad-CAM 热力图可视化模型关注区域
- **交互式界面**：Gradio Web 应用，支持图片上传与实时推理

## 数据集

数据集来源于 PlantVillage，共包含 **10 个类别**（9 种病害 + 健康）。
预期目录结构如下：

```text
tomato/
  train/
    Tomato___Bacterial_spot/         # 细菌性斑点
    Tomato___Early_blight/           # 早疫病
    Tomato___Late_blight/            # 晚疫病
    Tomato___Leaf_Mold/              # 叶霉病
    Tomato___Septoria_leaf_spot/     # 斑枯病
    Tomato___Spider_mites Two-spotted_spider_mite/  # 红蜘蛛
    Tomato___Target_Spot/            # 靶斑病
    Tomato___Tomato_mosaic_virus/    # 烟草花叶病毒
    Tomato___Tomato_Yellow_Leaf_Curl_Virus/  # 番茄黄化曲叶病毒
    Tomato___healthy/                # 健康
  val/
    (与 train 目录类别一致)
```

`train` 目录用于训练，`val` 目录以可复现方式按比例拆分为验证集和测试集。

## 项目结构

```
D:\桌面\人机A\大作业\
├── configs/
│   └── experiment_matrix.json      # 实验矩阵配置
├── outputs/
│   └── experiments/                # 各实验输出
├── src/
│   ├── __init__.py
│   ├── config.py                   # 实验配置与预设
│   ├── data.py                     # 数据加载与增强
│   ├── models.py                   # 模型构建与冻结策略
│   ├── trainer.py                  # 两阶段训练流程
│   ├── evaluator.py                # 评估指标与可视化
│   ├── gradcam.py                  # Grad-CAM 可视化
│   └── utils.py                    # 工具函数
├── tomato/
│   ├── train/                      # 训练集（10 类）
│   └── val/                        # 验证/测试集（10 类）
├── app.py                          # Gradio Web 应用
├── train.py                        # 训练入口
├── evaluate.py                     # 评估入口
├── summarize_experiments.py        # 实验汇总与对比图表
├── main.py                         # 整合版（训练+推理+Gradio）
├── main1.py                        # 完整整合版（旧版）
├── requirements.txt                # Python 依赖
└── 开题报告-郑喻文-3250103102.md/pdf  # 开题报告文档
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

使用预设实验进行训练（推荐先运行基线）：

```powershell
python -X utf8 train.py --preset resnet18_transfer_fc_then_layer4
```

### 评估模型

```powershell
python -X utf8 evaluate.py --run-dir outputs/experiments/resnet18_transfer_fc_then_layer4
```

### 启动 Gradio 交互界面

```powershell
python -X utf8 app.py --run-dir outputs/experiments/mobilenet_v3_transfer_fc_then_features_tail
```

### 汇总对比实验

```powershell
python -X utf8 summarize_experiments.py
```

## 实验预设

| 预设名称 | 架构 | 预训练 | 冻结策略 | 阶段1 LR | 阶段2 LR | 说明 |
|----------|------|--------|----------|---------|---------|------|
| `resnet18_transfer_fc_then_layer4` | ResNet-18 | 是 | fc->layer4 | 1e-3 | 1e-5 | **基线**：标准迁移学习 |
| `resnet18_scratch` | ResNet-18 | 否 | 无 | — | 1e-3 | 从零训练对照 |
| `resnet18_transfer_fc_only` | ResNet-18 | 是 | 仅分类头 | 1e-3 | — | 仅微调分类头 |
| `resnet18_transfer_lr_low` | ResNet-18 | 是 | fc->layer4 | 3e-4 | 1e-5 | 低学习率对照 |
| `resnet50_transfer_fc_then_layer4` | ResNet-50 | 是 | fc->layer4 | 1e-3 | 1e-5 | 更大骨干网络 |
| `mobilenet_v3_transfer_fc_then_features_tail` | MobileNet-V3 Small | 是 | fc->features | 1e-3 | 1e-5 | 轻量级网络 |

## 实验结果

所有实验均满足作业要求的测试 Top-1 准确率 >= 90%。

| 实验 | Top-1 准确率 | Macro F1 | 参数量 | 推理时间/张 |
|------|:-----------:|:--------:|:------:|:----------:|
| **MobileNet-V3 迁移学习** | **0.966** | **0.965** | **1.53M** | **0.000195s** |
| ResNet-50 迁移学习 | 0.962 | 0.962 | 23.53M | 0.002033s |
| ResNet-18 迁移学习(低学习率) | 0.958 | 0.958 | 11.18M | 0.000753s |
| ResNet-18 迁移学习(基线) | 0.954 | 0.954 | 11.18M | 0.000571s |
| ResNet-18 从零训练 | 0.930 | 0.927 | 11.18M | 0.000734s |
| ResNet-18 仅微调分类头 | 0.890 | 0.889 | 11.18M | 0.000739s |

**推荐模型**：MobileNet-V3 Small 迁移学习，以最小参数量取得最高准确率和最快推理速度。

## 技术细节

### 两阶段训练策略

1. **阶段1**：冻结骨干网络，仅训练分类头（高学习率 1e-3）
2. **阶段2**：解冻深层（layer4/features tail），联合微调（低学习率 1e-5）

### 数据增强

- 训练集：缩放至 224x224 -> 随机水平翻转 (p=0.5) -> 随机旋转 (+-15度) -> ToTensor -> ImageNet 标准化
- 验证/测试集：缩放至 224x224 -> ToTensor -> ImageNet 标准化

### Grad-CAM

自动为每轮实验生成 Grad-CAM 热力图，可视化模型在分类时关注的叶片区域，辅助判断模型学习到的特征是否合理。

### 输出产物

每轮实验在 `outputs/experiments/<名称>/` 下生成以下文件：

- `config.json` — 实验配置
- `best_model.pth` — 最佳模型权重
- `history.json` + `training_curves.png` — 训练过程记录与曲线
- `classification_report.txt/csv` — 详细分类报告
- `confusion_matrix.png` — 混淆矩阵
- `metrics.json` — 汇总指标（准确率、F1、推理速度）
- `grad_cam.png` — Grad-CAM 热力图

## 依赖

- Python >= 3.8
- PyTorch & torchvision
- NumPy / Matplotlib / Seaborn
- scikit-learn
- Gradio
- Pillow

完整依赖见 [requirements.txt](./requirements.txt)。

---

> 作者：郑喻文 | 学号：3250103102 | 人工智能基础(A) 课程大作业
