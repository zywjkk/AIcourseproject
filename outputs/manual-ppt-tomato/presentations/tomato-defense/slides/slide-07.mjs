import { base, title, C, smallTable, bar } from "./common.mjs";

export function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 7);
  title(slide, ctx, "实验矩阵与关键结果", "覆盖 Scratch、冻结全部、冻结前 N 层、学习率和 3 种骨干网络对比");
  smallTable(
    slide,
    ctx,
    ["实验", "骨干", "冻结策略", "Acc", "参数量"],
    [
      ["MobileNetV3", "small", "fc_then_layer4", "96.6%", "1.53M"],
      ["ResNet50", "resnet50", "fc_then_layer4", "96.2%", "23.53M"],
      ["ResNet18 lr=3e-4", "resnet18", "fc_then_layer4", "95.8%", "11.18M"],
      ["ResNet18 lr=1e-3", "resnet18", "fc_then_layer4", "95.4%", "11.18M"],
      ["ResNet18 Scratch", "resnet18", "none", "93.0%", "11.18M"],
      ["ResNet18 fc_only", "resnet18", "fc_only", "89.0%", "11.18M"],
    ],
    60,
    168,
    [235, 150, 230, 90, 110],
    35,
  );
  ctx.addText(slide, { text: "Accuracy Ranking", left: 875, top: 168, width: 190, height: 24, fontSize: 14, color: C.green, bold: true, typeface: "Aptos" });
  bar(slide, ctx, "MobileNetV3", 0.966, 1.0, 875, 210, 220, C.green);
  bar(slide, ctx, "ResNet50", 0.962, 1.0, 875, 248, 220, C.green2);
  bar(slide, ctx, "ResNet18 low LR", 0.958, 1.0, 875, 286, 220, C.green2);
  bar(slide, ctx, "ResNet18", 0.954, 1.0, 875, 324, 220, C.green2);
  bar(slide, ctx, "Scratch", 0.93, 1.0, 875, 362, 220, C.gold);
  bar(slide, ctx, "fc_only", 0.89, 1.0, 875, 400, 220, C.red);
  ctx.addText(slide, {
    text: "最终选择 MobileNetV3-small：准确率最高、参数量最低、单张推理最快，适合课程项目中的轻量部署演示。",
    left: 60,
    top: 550,
    width: 1040,
    height: 32,
    fontSize: 17,
    color: C.ink,
    bold: true,
    typeface: "Microsoft YaHei",
  });
  return slide;
}
