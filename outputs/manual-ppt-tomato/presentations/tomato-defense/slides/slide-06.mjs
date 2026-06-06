import { base, title, C, miniNode, bulletList } from "./common.mjs";

export function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 6);
  title(slide, ctx, "工程实现结构", "按 PyTorch 项目组织拆分，训练、评估、部署互相解耦");
  const nodes = [
    ["src/config.py\n实验配置 / seed", 90, 190],
    ["src/data.py\nDataset / DataLoader", 330, 190],
    ["src/models.py\nCNN / Backbone / Freeze", 570, 190],
    ["src/trainer.py\n训练循环 / 验证", 810, 190],
    ["src/evaluator.py\n报告 / 混淆矩阵", 210, 330],
    ["src/gradcam.py\n可解释性热力图", 450, 330],
    ["train.py / evaluate.py\n命令行入口", 690, 330],
    ["app.py\nGradio Top-3", 930, 330],
  ];
  nodes.forEach(([t, x, y]) => miniNode(slide, ctx, t, x, y, 185, 64, C.white));
  ctx.addText(slide, { text: "核心工程原则", left: 90, top: 498, width: 160, height: 22, fontSize: 14, color: C.green, bold: true, typeface: "Microsoft YaHei" });
  bulletList(
    slide,
    ctx,
    [
      "训练脚本可复现：固定随机种子、保存 config/history/metrics/model",
      "评估脚本独立运行：测试集分类报告、混淆矩阵、推理速度统计",
      "Web 演示复用同一 checkpoint 与 transform，避免训练/部署预处理不一致",
    ],
    90,
    532,
    900,
    32,
    14,
  );
  return slide;
}
