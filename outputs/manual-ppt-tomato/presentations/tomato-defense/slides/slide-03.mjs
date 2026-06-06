import { base, title, C, metric, bulletList } from "./common.mjs";

export function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 3);
  title(slide, ctx, "数据集与预处理", "PlantVillage Tomato：10 类病害/健康叶片，固定随机种子确保复现");
  metric(slide, ctx, "10000", "Train", 70, 185, 160);
  metric(slide, ctx, "500", "Validation", 250, 185, 160);
  metric(slide, ctx, "500", "Test", 430, 185, 160);
  ctx.addText(slide, {
    text: "划分策略",
    left: 70,
    top: 296,
    width: 180,
    height: 24,
    fontSize: 14,
    color: C.green,
    bold: true,
    typeface: "Microsoft YaHei",
  });
  bulletList(
    slide,
    ctx,
    [
      "原始 train 作为训练集；原始 val 按固定 seed 1:1 拆分为验证集与测试集",
      "固定 Python / NumPy / PyTorch 随机种子，报告中同步说明复现设置",
      "类别顺序由 ImageFolder 读取并保存，避免预测标签与训练标签错位",
    ],
    70,
    332,
    560,
    36,
    14,
  );
  ctx.addShape(slide, { left: 725, top: 180, width: 420, height: 310, fill: C.white, line: { fill: C.faint, width: 1, style: "solid" } });
  ctx.addText(slide, { text: "Transform Pipeline", left: 750, top: 205, width: 220, height: 24, fontSize: 15, bold: true, color: C.ink, typeface: "Aptos" });
  bulletList(
    slide,
    ctx,
    [
      { text: "Resize: 224 × 224", bold: true },
      { text: "RandomHorizontalFlip", bold: true },
      { text: "RandomRotation: ±15°", bold: true },
      { text: "ToTensor + ImageNet Normalize", bold: true },
      { text: "Valid/Test 仅做确定性预处理", muted: true },
    ],
    750,
    252,
    350,
    42,
    15,
  );
  return slide;
}
