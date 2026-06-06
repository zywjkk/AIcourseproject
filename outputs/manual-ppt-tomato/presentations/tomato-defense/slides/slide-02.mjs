import { base, title, C, smallTable } from "./common.mjs";

export function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 2);
  title(slide, ctx, "课程要求对齐", "把评分点转化为可检查的代码、实验和演示证据");
  smallTable(
    slide,
    ctx,
    ["课程要求", "项目实现", "证据文件/结果"],
    [
      ["完整训练脚本", "Dataset/DataLoader、模型、训练循环、验证评估", "train.py + src/*"],
      ["Top-1 ≥ 90%", "最佳 MobileNetV3-small 迁移学习", "96.6%"],
      ["CNN 与 Shape", "从输入到分类头的 Tensor Shape 记录", "model_architecture.txt"],
      ["迁移学习冻结", "fc_only / fc_then_layer4 / scratch 对比", "6 组实验"],
      ["评估可视化", "Loss/Acc、混淆矩阵、分类报告", "outputs/experiments/*"],
      ["进阶挑战", "Gradio Top-3 预测；3 骨干对比", "app.py + 对比图"],
    ],
    70,
    176,
    [230, 430, 330],
    43,
  );
  ctx.addShape(slide, { left: 70, top: 500, width: 990, height: 1, fill: C.faint });
  ctx.addText(slide, {
    text: "交付逻辑：源码负责可复现，实验报告负责可解释，PPT/演示视频负责可展示。",
    left: 70,
    top: 534,
    width: 910,
    height: 30,
    fontSize: 18,
    bold: true,
    color: C.green,
    typeface: "Microsoft YaHei",
  });
  return slide;
}
