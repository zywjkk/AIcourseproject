import { base, title, C, bulletList } from "./common.mjs";

export async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 9);
  title(slide, ctx, "可解释性与交互演示", "Grad-CAM 验证模型关注叶片病害区域；Gradio 支持上传图片实时 Top-3 预测");
  await ctx.addImage(slide, {
    path: `${ctx.workspaceDir}/../../../../experiments/mobilenet_v3_transfer_fc_then_features_tail/grad_cam.png`,
    left: 70,
    top: 172,
    width: 430,
    height: 320,
    fit: "contain",
    alt: "Grad-CAM",
  });
  ctx.addShape(slide, { left: 570, top: 178, width: 520, height: 286, fill: C.white, line: { fill: C.faint, width: 1, style: "solid" } });
  ctx.addText(slide, { text: "Gradio Demo 验证", left: 598, top: 208, width: 250, height: 28, fontSize: 18, bold: true, color: C.green, typeface: "Microsoft YaHei" });
  bulletList(
    slide,
    ctx,
    [
      "上传番茄叶片图片，返回 Top-3 类别和置信度",
      "Healthy 样本预测：Tomato___healthy，置信度 0.999996",
      "Yellow Leaf Curl Virus 样本预测正确，置信度 0.999882",
      "部署脚本 app.py 复用训练时的类别映射与图像预处理",
    ],
    598,
    258,
    430,
    44,
    15,
  );
  ctx.addText(slide, {
    text: "意义：不仅能给出分类结果，还能解释模型依据，便于答辩中说明模型不是随机猜测。",
    left: 70,
    top: 548,
    width: 990,
    height: 30,
    fontSize: 16,
    color: C.ink,
    bold: true,
    typeface: "Microsoft YaHei",
  });
  return slide;
}
