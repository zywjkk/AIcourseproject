# 番茄病害识别检测系统

## Tomato Disease Recognition System

**人工智能基础(A) · 选题四 · 第七组**

---

## 项目简介

基于 PyTorch 的番茄叶片病害图像分类系统。使用 PlantVillage 公开数据集，对 10 类番茄叶片（9 种病害 + 健康）进行分类。采用两阶段迁移学习策略，对比了 ResNet-18、ResNet-50、MobileNet-V3 Small 三种骨干网络。

**最佳结果：MobileNet-V3 Small 迁移学习，测试集 Top-1 准确率 96.6%。**

---

## 组员信息

| 姓名 | 学号 | 角色 |
|------|------|------|
| 郑喻文 | 3250103102 | 组长 |
| 方鑫 | | 组员 |
| 王骏扬 | | 组员 |
| 李军 | | 组员 |
| 王安卓 | | 组员 |

---

## 目录结构

```
├── src/                  核心源代码（7 个模块）
│   ├── config.py         实验配置与预设（6 组实验）
│   ├── data.py           数据加载与增强
│   ├── models.py         模型构建与冻结策略
│   ├── trainer.py        两阶段训练流程
│   ├── evaluator.py      评估指标与可视化
│   ├── gradcam.py        Grad-CAM 可解释性
│   └── utils.py          工具函数
├── web/                  专业版 Web 前端（FastAPI + SPA）
│   ├── app.py            FastAPI 后端（REST API）
│   └── static/           前端界面（HTML/CSS/JS + Chart.js）
├── train.py              训练入口脚本
├── evaluate.py            评估入口脚本
├── app.py                备用 Web 前端（Gradio）
├── summarize_experiments.py  实验汇总与对比图表生成
├── presentation/         答辩 PPT 材料
│   ├── guizang-web-deck/ 网页版 PPT（Swiss Design）
│   └── powerpoint/       传统 PowerPoint 版本
├── outputs/experiments/  实验输出结果
│   ├── model_architecture.txt   模型架构与参数量
│   ├── training_curves.png      训练 Loss/Accuracy 曲线
│   ├── confusion_matrix.png     测试集混淆矩阵
│   ├── grad_cam.png             Grad-CAM 热力图
│   └── metrics.json             准确率、F1、推理速度等指标
├── resnet18_tomato_structure    模型计算图（PyTorch autograd）
├── 网图测试用/           各病害示例图片（20 张，用于演示）
├── 实验报告-番茄病害识别检测系统.md     详细实验报告
├── 实验报告-番茄病害识别检测系统.docx    实验报告（Word 版）
├── 答辩PPT-番茄病害识别检测系统.pptx    答辩幻灯片
├── 开题报告-郑喻文-3250103102.md/pdf  开题报告
├── 课堂展示指导手册.md         答辩演示操作指南
├── requirements.txt        Python 依赖
└── .gitignore
```

---

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
# （可选）如需运行专业版 Web 前端，额外安装：
pip install fastapi uvicorn python-multipart
```

### 2. 使用预训练模型启动 Web 演示

模型权重已保存在 `outputs/experiments/` 下，**无需重新训练**即可直接启动。

**方式一（推荐）：专业版 Web 前端**

```powershell
cd web
python app.py --run-dir ../outputs/experiments/mobilenet_v3_transfer_fc_then_features_tail --port 8000
```

打开浏览器访问 `http://127.0.0.1:8000`

**方式二（备用）：Gradio 简易版**

```powershell
python -X utf8 app.py --run-dir outputs/experiments/mobilenet_v3_transfer_fc_then_features_tail
```

打开浏览器访问 `http://127.0.0.1:7860`

### 3. 测试图片

`网图测试用/` 目录下有 20 张不同病害的示例图片（每类 2 张），可直接上传测试。

---

## 实验预设

通过 `--preset` 参数选择实验：

| 预设名称 | 骨干网络 | 策略 | 准确率 |
|----------|----------|------|:------:|
| `mobilenet_v3_transfer_fc_then_features_tail` | MobileNet-V3 Small | 迁移学习 | **96.6%** |
| `resnet50_transfer_fc_then_layer4` | ResNet-50 | 迁移学习 | 96.2% |
| `resnet18_transfer_fc_then_layer4` | ResNet-18 | 迁移学习（基线） | 95.4% |
| `resnet18_transfer_lr_low` | ResNet-18 | 迁移学习（低学习率） | 95.8% |
| `resnet18_scratch` | ResNet-18 | 从零训练 | 93.0% |
| `resnet18_transfer_fc_only` | ResNet-18 | 仅微调分类头 | 89.0% |

---

## 核心结论

1. **迁移学习有效**：比从零训练高 2.4 个百分点
2. **解冻微调关键**：解冻后部层微调（95.4%）远优于完全冻结（89.0%）
3. **MobileNet-V3 最优**：1.53M 参数量 + 96.6% 准确率，最佳性价比
4. **Grad-CAM 可视化**：热力图显示模型聚焦叶片病斑区域

---

## 评分点对应

| 课程要求 | 对应位置 | 完成情况 |
|----------|----------|:--------:|
| 完整训练脚本 | `train.py` + `src/` | ✅ |
| 测试集 Top-1 ≥ 90% | 最高 96.6% | ✅ |
| 模型架构图 + Tensor Shape | `model_architecture.txt` | ✅ |
| 迁移学习（冻结策略对比） | `src/models.py` + 6 组实验 | ✅ |
| Loss/Accuracy 曲线 | `training_curves.png` | ✅ |
| 混淆矩阵 + 分类报告 | `confusion_matrix.png` + `classification_report.txt` | ✅ |
| 超参数对比 | 6 组实验（学习率/骨干网络/冻结策略） | ✅ |
| 固定随机种子 | `utils.py` seed=42 | ✅ |
| 模块化代码 | `src/` 下 7 个独立模块 | ✅ |
| **进阶：Web 界面** | `web/` FastAPI + SPA + Gradio 备用 | ✅ |
| **进阶：多骨干网络对比** | ResNet-18/50 + MobileNet-V3 | ✅ |
| **进阶：Grad-CAM** | `grad_cam.png` 每轮实验均有 | ✅ |

---

## 详细文档

- [实验报告](实验报告-番茄病害识别检测系统.md) — 完整的实验原理、结果分析和结论
- [课堂展示指导手册](课堂展示指导手册.md) — 答辩流程、Web 端操作、核心原理讲解
- [开题报告](开题报告-郑喻文-3250103102.md) — 选题调研和技术预研

---

## 技术栈

Python 3.13 · PyTorch 2.11 · torchvision · FastAPI · Gradio · Chart.js · Matplotlib · Seaborn · scikit-learn
