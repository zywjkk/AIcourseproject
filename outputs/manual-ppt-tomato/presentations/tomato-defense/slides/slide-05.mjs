import { base, title, C, smallTable } from "./common.mjs";

export function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 5);
  title(slide, ctx, "迁移学习与冻结策略", "冻结不是装饰项：它决定哪些 ImageNet 特征被保留，哪些层适配番茄病害");
  smallTable(
    slide,
    ctx,
    ["策略", "训练参数", "目的", "测试准确率"],
    [
      ["From Scratch", "全模型", "观察无预训练基线", "93.0%"],
      ["fc_only", "仅最后分类层", "验证“冻结全部 backbone”", "89.0%"],
      ["fc_then_layer4", "分类层 + 尾部特征层", "冻结前面 N 层，微调高层语义", "95.4%~96.6%"],
    ],
    70,
    180,
    [210, 230, 390, 130],
    52,
  );
  ctx.addShape(slide, { left: 106, top: 430, width: 840, height: 72, fill: C.white, line: { fill: C.faint, width: 1, style: "solid" } });
  ctx.addText(slide, { text: "冻结前面 N 层", left: 130, top: 454, width: 140, height: 26, fontSize: 14, bold: true, color: C.green, typeface: "Microsoft YaHei" });
  ctx.addShape(slide, { left: 290, top: 452, width: 150, height: 26, fill: C.faint });
  ctx.addShape(slide, { left: 450, top: 452, width: 150, height: 26, fill: C.faint });
  ctx.addShape(slide, { left: 610, top: 452, width: 150, height: 26, fill: C.pale });
  ctx.addShape(slide, { left: 770, top: 452, width: 120, height: 26, fill: C.green });
  ctx.addText(slide, { text: "浅层纹理", left: 312, top: 458, width: 100, height: 16, fontSize: 9, color: C.muted, align: "center", typeface: "Microsoft YaHei" });
  ctx.addText(slide, { text: "中层形状", left: 472, top: 458, width: 100, height: 16, fontSize: 9, color: C.muted, align: "center", typeface: "Microsoft YaHei" });
  ctx.addText(slide, { text: "高层语义", left: 632, top: 458, width: 100, height: 16, fontSize: 9, color: C.green, align: "center", typeface: "Microsoft YaHei" });
  ctx.addText(slide, { text: "分类头", left: 790, top: 458, width: 80, height: 16, fontSize: 9, color: C.white, align: "center", typeface: "Microsoft YaHei" });
  ctx.addText(slide, {
    text: "结论：仅训练分类头不足以适配病害细粒度差异；解冻尾部特征层后，迁移学习明显优于从头训练和 fc_only。",
    left: 106,
    top: 540,
    width: 880,
    height: 34,
    fontSize: 16,
    bold: true,
    color: C.ink,
    typeface: "Microsoft YaHei",
  });
  return slide;
}
