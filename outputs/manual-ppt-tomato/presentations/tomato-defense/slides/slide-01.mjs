import { base, C, metric } from "./common.mjs";

export function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 1, "人工智能基础(A) / 选题四");
  ctx.addShape(slide, { left: 50, top: 86, width: 7, height: 152, fill: C.green });
  ctx.addText(slide, {
    text: "番茄病害识别检测系统",
    left: 78,
    top: 82,
    width: 880,
    height: 54,
    fontSize: 34,
    color: C.ink,
    bold: true,
    typeface: "Microsoft YaHei",
  });
  ctx.addText(slide, {
    text: "深度学习在农业中的应用 / CNN + Transfer Learning + Gradio Demo",
    left: 80,
    top: 142,
    width: 840,
    height: 28,
    fontSize: 15,
    color: C.muted,
    typeface: "Aptos",
  });
  ctx.addText(slide, {
    text: "PlantVillage 番茄叶片 10 类分类任务；完整覆盖数据预处理、模型训练、迁移学习、评估可视化与交互部署。",
    left: 80,
    top: 204,
    width: 690,
    height: 58,
    fontSize: 17,
    color: C.ink,
    typeface: "Microsoft YaHei",
  });
  metric(slide, ctx, "96.6%", "测试集 Top-1 Accuracy", 80, 330, 190);
  metric(slide, ctx, "10", "番茄病害/健康类别", 300, 330, 190);
  metric(slide, ctx, "11k", "PlantVillage 图像规模", 520, 330, 190);
  metric(slide, ctx, "Top-3", "网页实时预测输出", 740, 330, 190);
  ctx.addShape(slide, { left: 80, top: 560, width: 1030, height: 1, fill: C.faint });
  ctx.addText(slide, {
    text: "答辩重点：核心原理与算法实现；不展开宏观背景，直接展示工程流程和实验结果。",
    left: 80,
    top: 584,
    width: 920,
    height: 26,
    fontSize: 13,
    color: C.muted,
    typeface: "Microsoft YaHei",
  });
  return slide;
}
