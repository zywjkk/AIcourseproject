import argparse
import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import build_transforms
from src.models import build_model
from src.utils import load_json

# ---------------------------------------------------------------------------
# 中文病害名称映射
# ---------------------------------------------------------------------------
DISEASE_DICT = {
    "Tomato___Bacterial_spot": "\u7ec6\u83cc\u6027\u6591\u70b9\u75c5",
    "Tomato___Early_blight": "\u65e9\u75ab\u75c5",
    "Tomato___Late_blight": "\u665a\u75ab\u75c5",
    "Tomato___Leaf_Mold": "\u53f6\u9709\u75c5",
    "Tomato___Septoria_leaf_spot": "\u6591\u67af\u75c5",
    "Tomato___Spider_mites Two-spotted_spider_mite": "\u4e8c\u6591\u53f6\u87a8",
    "Tomato___Target_Spot": "\u9776\u6591\u75c5",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "\u9ec4\u5316\u5377\u53f6\u75c5\u6bd2\u75c5",
    "Tomato___Tomato_mosaic_virus": "\u82b1\u53f6\u75c5\u6bd2\u75c5",
    "Tomato___healthy": "\u5065\u5eb7\u53f6\u7247",
}


def get_cn_classes(classes):
    return [DISEASE_DICT.get(c, c.replace("Tomato___", "").replace("_", " ")) for c in classes]


# ---------------------------------------------------------------------------
# 自定义 CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --primary: #e85d4a;
    --primary-dark: #c94a3a;
    --primary-light: #fff0ee;
    --accent: #4caf7d;
    --accent-light: #e8f5e9;
    --bg-warm: #faf8f5;
    --text-main: #2d2a26;
    --text-muted: #6b6560;
    --border: #e8e2dc;
    --card-bg: #ffffff;
    --radius: 16px;
    --shadow: 0 4px 24px rgba(45, 42, 38, 0.08);
    --shadow-hover: 0 8px 40px rgba(45, 42, 38, 0.12);
}
body { background: var(--bg-warm) !important; font-family: 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', sans-serif !important; }
.gradio-container { background: var(--bg-warm) !important; }
.header-bar { background: linear-gradient(135deg, #e85d4a 0%, #f07a6a 50%, #f5a623 100%) !important; border-radius: var(--radius) !important; padding: 36px 40px !important; margin-bottom: 24px !important; box-shadow: var(--shadow-hover) !important; position: relative !important; overflow: hidden !important; }
.header-bar::before { content: ''; position: absolute; top: -50%; right: -10%; width: 300px; height: 300px; background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%); border-radius: 50%; }
.header-bar::after { content: ''; position: absolute; bottom: -30%; left: -5%; width: 200px; height: 200px; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); border-radius: 50%; }
.header-title { color: #ffffff !important; font-size: 2.2em !important; font-weight: 700 !important; margin-bottom: 8px !important; position: relative !important; z-index: 1 !important; }
.header-subtitle { color: rgba(255,255,255,0.92) !important; font-size: 1.05em !important; position: relative !important; z-index: 1 !important; }
.header-badge { display: inline-flex !important; align-items: center !important; background: rgba(255,255,255,0.2) !important; backdrop-filter: blur(10px) !important; border: 1px solid rgba(255,255,255,0.3) !important; color: #fff !important; padding: 6px 14px !important; border-radius: 20px !important; font-size: 0.85em !important; margin-top: 12px !important; position: relative !important; z-index: 1 !important; }
.info-section { background: var(--card-bg) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 20px 24px !important; margin-bottom: 20px !important; box-shadow: var(--shadow) !important; }
.info-title { color: var(--text-main) !important; font-size: 1.1em !important; font-weight: 600 !important; margin-bottom: 12px !important; display: flex !important; align-items: center !important; gap: 8px !important; }
.disease-grid { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)) !important; gap: 8px !important; }
.disease-chip { background: var(--primary-light) !important; color: var(--primary-dark) !important; padding: 6px 12px !important; border-radius: 8px !important; font-size: 0.82em !important; font-weight: 500 !important; border-left: 3px solid var(--primary) !important; }
.disease-chip:hover { background: var(--primary) !important; color: #fff !important; transform: translateY(-1px) !important; }
.upload-card { background: var(--card-bg) !important; border: 2px dashed var(--border) !important; border-radius: var(--radius) !important; box-shadow: var(--shadow) !important; transition: all 0.3s ease !important; }
.upload-card:hover { border-color: var(--primary) !important; box-shadow: var(--shadow-hover) !important; }
.result-card { background: var(--card-bg) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 24px !important; box-shadow: var(--shadow) !important; }
.footer-bar { background: var(--card-bg) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 20px 28px !important; margin-top: 20px !important; box-shadow: var(--shadow) !important; }
.footer-content { display: flex !important; flex-wrap: wrap !important; gap: 20px !important; justify-content: space-between !important; align-items: center !important; }
.footer-item { display: flex !important; align-items: center !important; gap: 8px !important; color: var(--text-muted) !important; font-size: 0.9em !important; }
.footer-dot { width: 8px !important; height: 8px !important; border-radius: 50% !important; background: var(--accent) !important; }
.gr-button { background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important; color: #fff !important; border: none !important; border-radius: 12px !important; padding: 10px 28px !important; font-weight: 600 !important; box-shadow: 0 4px 12px rgba(232,93,74,0.3) !important; }
.gr-button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(232,93,74,0.4) !important; }
@media (max-width: 768px) { .header-title { font-size: 1.5em !important; } .disease-grid { grid-template-columns: 1fr 1fr !important; } .footer-content { flex-direction: column !important; align-items: flex-start !important; } }
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio tomato disease detector.")
    parser.add_argument("--run-dir", required=True, help="Experiment directory with best_model.pth.")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / "config.json")
    dataset_summary = load_json(run_dir / "dataset_summary.json")
    classes = dataset_summary["classes"]

    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    device = torch.device("cpu")
    model = build_model(config["architecture"], len(classes), False)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    model.eval()
    _, eval_transform = build_transforms(config["image_size"])

    cn_classes = get_cn_classes(classes)

    def predict_image(image):
        if image is None:
            return None
        tensor = eval_transform(image).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor)[0], dim=0)
        return {cn_classes[i]: float(probs[i]) for i in range(len(classes))}

    def clear_all():
        return None, None

    disease_chips = "".join(f'<div class="disease-chip">{c}</div>' for c in cn_classes)

    with gr.Blocks(css=CUSTOM_CSS, title="智慧农业：番茄病害识别检测系统") as app:
        gr.HTML('<div class="header-bar">'
                '<div class="header-title">番茄病害识别检测系统</div>'
                '<div class="header-subtitle">基于深度迁移学习训练，测试集准确率 &gt; 90%。<br>上传番茄叶片图像，系统将实时诊断并给出置信度分析。</div>'
                '<div class="header-badge">AI 驱动的智能诊断</div></div>')

        gr.HTML(f'<div class="info-section">'
                f'<div class="info-title">系统可识别的病害类别（共 {len(cn_classes)} 类）</div>'
                f'<div class="disease-grid">{disease_chips}</div></div>')

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.HTML('<div style="margin-bottom:8px;font-weight:600;color:#2d2a26;font-size:1.05em;">上传叶片图像</div>')
                image_input = gr.Image(type="pil", label="", height=380, elem_classes="upload-card")
                with gr.Row():
                    analyze_btn = gr.Button(value="开始智能诊断", variant="primary", size="lg")
                    clear_btn = gr.Button(value="清除重试", size="lg")

            with gr.Column(scale=5):
                gr.HTML('<div style="margin-bottom:8px;font-weight:600;color:#2d2a26;font-size:1.05em;">AI 诊断结果</div>')
                result_output = gr.Label(num_top_classes=len(classes), label="", elem_classes="result-card")

        gr.HTML('<div class="footer-bar"><div class="footer-content">'
                '<div class="footer-item"><div class="footer-dot"></div><span>模型架构：深度残差网络</span></div>'
                '<div class="footer-item"><div class="footer-dot"></div><span>训练策略：两阶段迁移学习</span></div>'
                '<div class="footer-item"><div class="footer-dot"></div><span>图像输入：224 x 224 像素</span></div>'
                '<div class="footer-item"><div class="footer-dot"></div><span>技术支持：PyTorch + Gradio</span></div>'
                '</div></div>')

        analyze_btn.click(fn=predict_image, inputs=image_input, outputs=result_output)
        clear_btn.click(fn=clear_all, inputs=[], outputs=[image_input, result_output])

    app.launch(share=False, server_name="127.0.0.1", server_port=args.server_port)


if __name__ == "__main__":
    main()
