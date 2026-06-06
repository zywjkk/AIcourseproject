import { base, title, C, miniNode, label } from "./common.mjs";

export function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 4);
  title(slide, ctx, "CNN 核心结构与 Tensor Shape", "从局部纹理到病害类别：卷积特征逐层抽象，分类头输出 10 维概率");
  const nodes = [
    ["Input\n[B,3,224,224]", 70, 220, 145],
    ["Conv / BN / ReLU\n[B,64,112,112]", 245, 220, 170],
    ["MaxPool + Blocks\n[B,64,56,56]", 450, 220, 165],
    ["Layer2-4\n[B,512,7,7]", 650, 220, 155],
    ["AvgPool + Linear\n[B,10]", 840, 220, 150],
  ];
  nodes.forEach(([t, x, y, w], i) => {
    miniNode(slide, ctx, t, x, y, w, 72, i === nodes.length - 1 ? C.pale : C.white);
    if (i < nodes.length - 1) {
      ctx.addShape(slide, { left: x + w + 15, top: y + 35, width: 30, height: 2, fill: C.green2 });
    }
  });
  label(slide, ctx, "关键组件", 70, 360);
  ctx.addText(slide, {
    text: "Conv2D 提取叶片斑点、边缘和颜色纹理；MaxPool 降低空间分辨率并增强平移鲁棒性；ReLU 引入非线性；Dropout/权重衰减缓解过拟合；Linear 将高层语义映射到 10 类。",
    left: 70,
    top: 392,
    width: 980,
    height: 70,
    fontSize: 17,
    color: C.ink,
    typeface: "Microsoft YaHei",
  });
  label(slide, ctx, "报告要求覆盖", 70, 514, 180, C.gold);
  ctx.addText(slide, {
    text: "实验报告中给出模型结构文本、参数量、Tensor Shape 变换，并保存每组实验的 model_architecture.txt。",
    left: 70,
    top: 548,
    width: 900,
    height: 30,
    fontSize: 15,
    color: C.muted,
    typeface: "Microsoft YaHei",
  });
  return slide;
}
