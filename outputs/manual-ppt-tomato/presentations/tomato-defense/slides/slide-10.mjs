import { base, title, C, bulletList, metric } from "./common.mjs";

export function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 10);
  title(slide, ctx, "结论与答辩要点", "项目严格覆盖课程要求，并完成进阶挑战中的交互界面与多骨干对比");
  metric(slide, ctx, "达成", "Top-1 ≥ 90%", 80, 182, 170);
  metric(slide, ctx, "完整", "训练-评估-部署流程", 280, 182, 200);
  metric(slide, ctx, "清晰", "模块化 PyTorch 结构", 510, 182, 200);
  metric(slide, ctx, "加分", "Gradio + 3 Backbone", 740, 182, 210);
  ctx.addText(slide, { text: "5分钟展示顺序", left: 86, top: 330, width: 180, height: 24, fontSize: 15, color: C.green, bold: true, typeface: "Microsoft YaHei" });
  bulletList(
    slide,
    ctx,
    [
      "先说明任务与要求对齐：10 类、90% 指标、不可 AutoML",
      "再讲核心方法：CNN Shape 流转、迁移学习冻结策略",
      "随后展示工程：模块拆分、训练脚本、评估脚本、Web Demo",
      "最后用结果收束：96.6% Accuracy、Macro-F1 96.5%、推理速度和 Grad-CAM",
    ],
    86,
    370,
    780,
    40,
    16,
  );
  ctx.addShape(slide, { left: 86, top: 574, width: 960, height: 1, fill: C.faint });
  ctx.addText(slide, {
    text: "一句话总结：该系统完成了番茄叶片病害识别从数据、模型、训练、评估到交互部署的完整计算机视觉工程闭环。",
    left: 86,
    top: 600,
    width: 940,
    height: 34,
    fontSize: 18,
    color: C.ink,
    bold: true,
    typeface: "Microsoft YaHei",
  });
  return slide;
}
