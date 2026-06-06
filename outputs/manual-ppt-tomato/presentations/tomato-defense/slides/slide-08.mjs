import { base, title, C, bulletList } from "./common.mjs";

export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 8);
  title(slide, ctx, "训练过程与测试集诊断", "Loss/Accuracy 曲线、混淆矩阵和分类报告共同证明模型效果");
  await ctx.addImage(slide, {
    path: `${ctx.workspaceDir}/../../../../experiments/mobilenet_v3_transfer_fc_then_features_tail/training_curves.png`,
    left: 62,
    top: 170,
    width: 500,
    height: 310,
    fit: "contain",
    alt: "Training curves",
  });
  await ctx.addImage(slide, {
    path: `${ctx.workspaceDir}/../../../../experiments/mobilenet_v3_transfer_fc_then_features_tail/confusion_matrix.png`,
    left: 610,
    top: 170,
    width: 420,
    height: 310,
    fit: "contain",
    alt: "Confusion matrix",
  });
  ctx.addShape(slide, { left: 1046, top: 174, width: 145, height: 302, fill: C.white, line: { fill: C.faint, width: 1, style: "solid" } });
  ctx.addText(slide, { text: "Best model", left: 1064, top: 198, width: 105, height: 22, fontSize: 13, color: C.green, bold: true, typeface: "Aptos" });
  bulletList(
    slide,
    ctx,
    [
      { text: "Top-1 96.6%", bold: true },
      { text: "Macro-F1 96.5%", bold: true },
      { text: "healthy F1=1.00", bold: true },
      { text: "Septoria F1=0.93", muted: true },
      { text: "500 张测试图", muted: true },
    ],
    1064,
    238,
    110,
    40,
    11,
  );
  ctx.addText(slide, {
    text: "测试集中主要误差集中在病斑纹理相近的类别；整体 Precision / Recall / F1 均达到约 0.97。",
    left: 62,
    top: 540,
    width: 960,
    height: 30,
    fontSize: 16,
    color: C.ink,
    typeface: "Microsoft YaHei",
  });
  return slide;
}
