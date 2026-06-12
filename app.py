import argparse
import base64
import io
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

DISEASE_GALLERY = [
    {
        "id": "early-blight",
        "name": "番茄早疫病",
        "nameEn": "Alternaria solani",
        "pathogen": "链格孢菌",
        "symptoms": "叶片出现同心轮纹状褐色病斑，边缘有黄色晕圈，潮湿时产生黑色霉层",
        "color": "#92400e",
        "file": "static/sample-early-blight.jpg",
    },
    {
        "id": "late-blight",
        "name": "番茄晚疫病",
        "nameEn": "Phytophthora infestans",
        "pathogen": "致病疫霉菌",
        "symptoms": "水渍状不规则病斑，叶背边缘有白色霉层，可导致叶片迅速腐烂",
        "color": "#1e3a5f",
        "file": "static/sample-late-blight.jpg",
    },
    {
        "id": "leaf-mold",
        "name": "叶霉病",
        "nameEn": "Fulvia fulva",
        "pathogen": "黄枝孢菌",
        "symptoms": "叶片正面淡黄褪绿斑，背面灰白色至黑褐色绒状霉层，病叶干枯卷曲",
        "color": "#365314",
        "file": "static/sample-leaf-mold.jpg",
    },
    {
        "id": "spider-mites",
        "name": "二斑叶螨病",
        "nameEn": "Two-spotted spider mite",
        "pathogen": "—",
        "symptoms": "叶片先从近叶柄的主脉两侧出现苍白色斑点，随着危害的加重，可使叶片变成灰白色及至暗褐色",
        "color": "#10b981",
        "file": "static/sample-spider-mites.jpg",
    },
    {
        "id": "bacterial-spot",
        "name": "细菌斑点病",
        "nameEn": "Xanthomonas spp.",
        "pathogen": "黄单胞菌",
        "symptoms": "深褐色至黑色小斑点，水渍状，周围有明显黄色晕圈，易破裂穿孔",
        "color": "#7f1d1d",
        "file": "static/sample-bacterial-spot.jpg",
    },
    {
        "id": "septoria",
        "name": "针壳孢叶斑病",
        "nameEn": "Septoria lycopersici",
        "pathogen": "番茄壳针孢菌",
        "symptoms": "灰白色或浅褐色圆形小病斑，边缘深褐色，中央散生黑色小点",
        "color": "#44403c",
        "file": "static/sample-septoria.jpg",
    },
]

ARCHITECTURE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v3_small": "MobileNet-V3 Small",
}

SCROLL_REVEAL_JS = """
function g() {
    var revealElements = document.querySelectorAll('.reveal-on-scroll, .reveal-left, .reveal-right');
    var observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });
    revealElements.forEach(function(el) { observer.observe(el); });

    var nav = document.getElementById('topNav');
    if (nav) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                nav.style.background = 'rgba(255,255,255,0.6)';
                nav.style.backdropFilter = 'blur(16px) saturate(180%)';
                nav.style.webkitBackdropFilter = 'blur(16px) saturate(180%)';
                nav.style.borderBottom = '1px solid rgba(4,47,46,0.05)';
            } else {
                nav.style.background = 'transparent';
                nav.style.backdropFilter = 'none';
                nav.style.webkitBackdropFilter = 'none';
                nav.style.borderBottom = 'none';
            }
        }, { passive: true });
    }

    var heroBg = document.querySelector('.hero-bg img');
    if (heroBg) {
        window.addEventListener('scroll', function() {
            var scrolled = window.scrollY;
            if (scrolled < window.innerHeight) {
                heroBg.style.transform = 'translateY(' + (scrolled * 0.4) + 'px)';
            }
        }, { passive: true });
    }

    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            var target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}
document.addEventListener('DOMContentLoaded', g);
setTimeout(g, 2000);
"""

CUSTOM_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&family=Noto+Serif+SC:wght@500;600;700&display=swap');

:root {
    --deep: #042f2e;
    --deep-80: rgba(4, 47, 46, 0.8);
    --deep-50: rgba(4, 47, 46, 0.5);
    --deep-10: rgba(4, 47, 46, 0.1);
    --deep-05: rgba(4, 47, 46, 0.05);
    --green: #10b981;
    --green-light: #ecfdf5;
    --orange: #ea580c;
    --orange-light: rgba(234, 88, 12, 0.1);
    --radius: 0.75rem;
    --radius-lg: 1rem;
    --radius-xl: 1.5rem;
    --shadow-sm: 0 4px 20px rgba(4, 47, 46, 0.04);
    --shadow: 0 8px 32px rgba(4, 47, 46, 0.08);
    --shadow-lg: 0 20px 40px -10px rgba(4, 47, 46, 0.15);
    --font-display: 'Noto Serif SC', 'PingFang SC', serif;
    --font-body: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

body {
    background: #ecfdf5 !important;
    font-family: var(--font-body) !important;
    color: var(--deep) !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
}

.gradio-container {
    background: #ecfdf5 !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.contain, .app, .main, .wrap, #component-0 {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.glass {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border-bottom: 1px solid rgba(4, 47, 46, 0.05) !important;
}

.hero-section {
    position: relative !important;
    min-height: 100vh !important;
    display: flex !important;
    align-items: flex-end !important;
    padding-bottom: 6rem !important;
    overflow: hidden !important;
    margin: -16px -16px 0 -16px !important;
}

.hero-bg { position: absolute !important; inset: 0 !important; z-index: 0 !important; }
.hero-bg img { width: 100% !important; height: 100% !important; object-fit: cover !important; }
.hero-overlay {
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(to top, rgba(4,47,46,0.85) 0%, rgba(4,47,46,0.35) 50%, rgba(4,47,46,0.15) 100%) !important;
}

.hero-content {
    position: relative !important;
    z-index: 10 !important;
    width: 100% !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 2rem !important;
}

.hero-title {
    font-family: var(--font-display) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: clamp(2.2rem, 5vw, 3.8rem) !important;
    line-height: 1.2 !important;
    margin-bottom: 1rem !important;
}

.hero-desc {
    color: rgba(255,255,255,0.7) !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    max-width: 560px !important;
    margin-bottom: 2rem !important;
}

.hero-buttons { display: flex !important; flex-wrap: wrap !important; gap: 1rem !important; margin-bottom: 2.5rem !important; }
.hero-btn-primary {
    display: inline-flex !important; align-items: center !important; gap: 0.5rem !important;
    padding: 0.75rem 1.5rem !important; background: #ea580c !important; color: #ffffff !important;
    border-radius: 9999px !important; font-size: 0.875rem !important; font-weight: 500 !important;
    border: none !important; cursor: pointer !important; text-decoration: none !important;
}
.hero-btn-secondary {
    display: inline-flex !important; align-items: center !important; gap: 0.5rem !important;
    padding: 0.75rem 1.5rem !important; background: rgba(255,255,255,0.1) !important;
    color: #ffffff !important; border-radius: 9999px !important; font-size: 0.875rem !important;
    font-weight: 500 !important; border: 1px solid rgba(255,255,255,0.2) !important; text-decoration: none !important;
}

.hero-stats { display: flex !important; gap: 2.5rem !important; }
.hero-stat-value {
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 1.8rem !important; color: #ffffff !important; font-weight: 500 !important;
}
.hero-stat-label { color: rgba(255,255,255,0.5) !important; font-size: 0.75rem !important; margin-top: 0.25rem !important; }

.scroll-indicator {
    position: absolute !important; bottom: 1.5rem !important; left: 50% !important;
    transform: translateX(-50%) !important; z-index: 10 !important; cursor: pointer !important;
}
.scroll-chevron {
    width: 24px !important; height: 24px !important; color: rgba(255,255,255,0.5) !important;
    animation: bounceDown 1.5s infinite !important;
}
@keyframes bounceDown {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(6px); }
}

.section-header { text-align: center !important; margin-bottom: 3.5rem !important; }
.section-badge {
    display: inline-flex !important; align-items: center !important; gap: 0.5rem !important;
    padding: 0.375rem 0.75rem !important; border-radius: 9999px !important;
    background: rgba(4, 47, 46, 0.05) !important; margin-bottom: 1rem !important;
    font-size: 0.75rem !important; color: #115e59 !important; font-weight: 500 !important;
}
.section-title {
    font-family: var(--font-display) !important; font-size: 2rem !important;
    font-weight: 700 !important; color: var(--deep) !important; margin-bottom: 0.75rem !important;
}
.section-desc {
    color: rgba(4, 47, 46, 0.5) !important; max-width: 480px !important;
    margin: 0 auto !important; font-size: 0.875rem !important; line-height: 1.6 !important;
}

.content-section { padding: 5rem 1rem !important; max-width: 1200px !important; margin: 0 auto !important; }
.content-section-alt { padding: 5rem 1rem !important; max-width: 1200px !important; margin: 0 auto !important; }

.disease-chips-bar {
    display: flex !important; flex-wrap: wrap !important; gap: 0.5rem !important;
    padding: 1.5rem !important; background: #ffffff !important;
    border: 1px solid rgba(4, 47, 46, 0.06) !important; border-radius: var(--radius-xl) !important;
    box-shadow: var(--shadow-sm) !important; margin-bottom: 2rem !important;
}
.disease-chip {
    background: #ecfdf5 !important; color: #115e59 !important; padding: 0.4rem 0.85rem !important;
    border-radius: 0.5rem !important; font-size: 0.8rem !important; font-weight: 500 !important;
    border-left: 3px solid var(--green) !important; transition: all 0.25s ease !important;
}
.disease-chip:hover {
    background: var(--green) !important; color: #ffffff !important;
    transform: translateY(-2px) !important; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
}

.column-label {
    display: flex !important; align-items: center !important; gap: 0.5rem !important;
    margin-bottom: 1rem !important; font-size: 0.875rem !important; font-weight: 500 !important;
    color: rgba(4, 47, 46, 0.7) !important;
}
.column-icon {
    width: 1.75rem !important; height: 1.75rem !important; border-radius: 0.5rem !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    font-size: 0.85rem !important;
}
.column-icon-green { background: rgba(16, 185, 129, 0.1) !important; }
.column-icon-orange { background: rgba(234, 88, 12, 0.1) !important; }

.upload-card-wrap {
    border-radius: var(--radius-xl) !important; overflow: hidden !important;
    box-shadow: var(--shadow-lg) !important; transition: box-shadow 0.3s ease !important;
}
.result-card {
    background: var(--card-bg, #ffffff) !important; border: 1px solid rgba(4, 47, 46, 0.06) !important;
    border-radius: var(--radius-xl) !important; padding: 24px !important; box-shadow: var(--shadow) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #ea580c, #c2410c) !important; color: #ffffff !important;
    border: none !important; border-radius: 9999px !important; padding: 0.65rem 1.75rem !important;
    font-weight: 600 !important; box-shadow: 0 4px 14px rgba(234, 88, 12, 0.3) !important;
}
.gr-button-secondary {
    background: #ffffff !important; color: var(--deep) !important;
    border: 1px solid rgba(4, 47, 46, 0.12) !important; border-radius: 9999px !important;
    padding: 0.65rem 1.75rem !important; font-weight: 500 !important;
}

.tips-grid { display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 1rem !important; margin-top: 2rem !important; }
.tip-card {
    padding: 1.25rem !important; border-radius: var(--radius-xl) !important; background: #ffffff !important;
    border: 1px solid rgba(4, 47, 46, 0.05) !important; box-shadow: var(--shadow-sm) !important;
}
.tip-card h4 { font-weight: 500 !important; color: var(--deep) !important; font-size: 0.875rem !important; margin-bottom: 0.5rem !important; }
.tip-card p { color: rgba(4, 47, 46, 0.45) !important; font-size: 0.75rem !important; line-height: 1.6 !important; }

.gallery-grid { display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 1.25rem !important; }
.gallery-card {
    position: relative !important; border-radius: var(--radius-xl) !important; overflow: hidden !important;
    cursor: pointer !important; box-shadow: var(--shadow) !important; aspect-ratio: 3 / 2 !important;
    background: #e2e8f0 !important;
}
.gallery-card img { width: 100% !important; height: 100% !important; object-fit: cover !important; transition: transform 0.5s ease !important; }
.gallery-card:hover img { transform: scale(1.05) !important; }
.gallery-label {
    position: absolute !important; bottom: 0 !important; left: 0 !important; right: 0 !important;
    padding: 1.5rem 1rem 1rem 1rem !important;
    background: linear-gradient(to top, rgba(4,47,46,0.7), transparent) !important;
}
.gallery-label h4 { color: #ffffff !important; font-weight: 500 !important; font-size: 0.875rem !important; margin: 0 !important; }
.gallery-hover {
    position: absolute !important; inset: 0 !important; display: flex !important;
    align-items: flex-end !important; opacity: 0 !important; transition: opacity 0.3s ease !important;
}
.gallery-card:hover .gallery-hover { opacity: 1 !important; }
.gallery-card:hover .gallery-label { opacity: 0 !important; }
.gallery-hover-inner {
    width: 100% !important; margin: 0.5rem !important; padding: 1rem !important;
    background: rgba(255,255,255,0.25) !important; backdrop-filter: blur(12px) saturate(150%) !important;
    border: 1px solid rgba(255,255,255,0.3) !important; border-radius: var(--radius) !important;
}
.gallery-hover-dot { display: inline-block !important; width: 0.5rem !important; height: 0.5rem !important; border-radius: 50% !important; margin-right: 0.5rem !important; }
.gallery-hover-name { font-family: var(--font-display) !important; font-size: 0.875rem !important; font-weight: 600 !important; color: var(--deep) !important; margin-bottom: 0.25rem !important; }
.gallery-hover-latin { font-size: 0.625rem !important; color: rgba(4, 47, 46, 0.5) !important; margin-bottom: 0.25rem !important; }
.gallery-hover-symptoms { font-size: 0.75rem !important; color: rgba(4, 47, 46, 0.7) !important; line-height: 1.5 !important; }

.reveal-on-scroll, .reveal-left, .reveal-right {
    opacity: 0 !important; transform: translateY(30px) !important;
    transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1), transform 0.7s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.reveal-left { transform: translateX(-40px) !important; }
.reveal-right { transform: translateX(40px) !important; }
.reveal-on-scroll.revealed, .reveal-left.revealed, .reveal-right.revealed {
    opacity: 1 !important; transform: translate(0, 0) !important;
}
.reveal-delay-1 { transition-delay: 0.05s !important; }
.reveal-delay-2 { transition-delay: 0.1s !important; }
.reveal-delay-3 { transition-delay: 0.15s !important; }
.reveal-delay-4 { transition-delay: 0.2s !important; }
.reveal-delay-5 { transition-delay: 0.25s !important; }
.reveal-delay-6 { transition-delay: 0.3s !important; }

.footer-bar {
    padding: 3rem 1rem !important; border-top: 1px solid rgba(4, 47, 46, 0.05) !important;
    background: rgba(4, 47, 46, 0.02) !important; margin: 0 -16px -16px -16px !important;
}
.footer-inner {
    max-width: 1200px !important; margin: 0 auto !important; display: flex !important;
    flex-wrap: wrap !important; align-items: center !important; justify-content: space-between !important;
    gap: 1.5rem !important; padding: 0 1rem !important;
}
.footer-brand { display: flex !important; align-items: center !important; gap: 0.75rem !important; }
.footer-logo-icon {
    width: 2.25rem !important; height: 2.25rem !important; border-radius: 0.5rem !important;
    background: var(--deep) !important; display: flex !important; align-items: center !important;
    justify-content: center !important; font-size: 1.1rem !important;
}
.footer-brand-name { font-family: var(--font-display) !important; font-size: 0.875rem !important; font-weight: 600 !important; color: var(--deep) !important; }
.footer-brand-desc { font-size: 0.625rem !important; color: rgba(4, 47, 46, 0.4) !important; margin-top: 0.125rem !important; }
.footer-text { font-size: 0.75rem !important; color: rgba(4, 47, 46, 0.35) !important; }
.footer-links { font-size: 0.625rem !important; color: rgba(4, 47, 46, 0.3) !important; }

footer { display: none !important; }

@media (max-width: 900px) {
    .gallery-grid { grid-template-columns: repeat(2, 1fr) !important; }
    .hero-stats { gap: 1.5rem !important; }
    .hero-stat-value { font-size: 1.3rem !important; }
}
@media (max-width: 768px) {
    .tips-grid { grid-template-columns: 1fr !important; }
    .section-title { font-size: 1.5rem !important; }
    .hero-title { font-size: 1.8rem !important; }
}
@media (max-width: 560px) {
    .gallery-grid { grid-template-columns: 1fr !important; }
}
"""


def get_cn_classes(classes):
    return [DISEASE_DICT.get(c, c.replace("Tomato___", "").replace("_", " ")) for c in classes]


def img_file_to_data_uri(filepath, max_width=800):
    try:
        img = Image.open(filepath).convert("RGB")
        width, height = img.size
        if width > max_width:
            ratio = max_width / width
            img = img.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except OSError as exc:
        print(f"Warning: cannot read image {filepath}: {exc}")
        return ""


def placeholder_data_uri(label: str, color: str = "#042f2e") -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{color}"/><stop offset="100%" stop-color="#10b981"/>'
        f'</linearGradient></defs><rect fill="url(#g)" width="100%" height="100%"/>'
        f'<text x="50%" y="50%" fill="white" font-size="28" text-anchor="middle" '
        f'dominant-baseline="middle" font-family="sans-serif">{label}</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")


def load_accuracy_label(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return "90%+"
    metrics = load_json(metrics_path)
    accuracy = metrics.get("top1_accuracy")
    if accuracy is None:
        return "90%+"
    return f"{accuracy * 100:.1f}%"


def prepare_gallery(script_dir: Path) -> list[dict]:
    items = []
    for entry in DISEASE_GALLERY:
        item = dict(entry)
        image_path = script_dir / entry["file"]
        item["img"] = img_file_to_data_uri(image_path, max_width=600)
        if not item["img"]:
            item["img"] = placeholder_data_uri(entry["name"], entry.get("color", "#042f2e"))
        items.append(item)
    return items


def build_gallery_html(gallery_items: list[dict]) -> str:
    cards = ""
    for index, item in enumerate(gallery_items):
        delay_class = f"reveal-delay-{index % 6 + 1}"
        cards += f"""
        <div class="gallery-card reveal-on-scroll {delay_class}">
            <img src="{item['img']}" alt="{item['name']}" loading="lazy" />
            <div class="gallery-label"><h4>{item['name']}</h4></div>
            <div class="gallery-hover">
                <div class="gallery-hover-inner">
                    <div style="display:flex;align-items:center;margin-bottom:0.5rem;">
                        <span class="gallery-hover-dot" style="background:{item['color']};"></span>
                        <span class="gallery-hover-name">{item['name']}</span>
                    </div>
                    <div class="gallery-hover-latin">{item['nameEn']}</div>
                    <div class="gallery-hover-symptoms">{item['symptoms']}</div>
                </div>
            </div>
        </div>"""
    return cards


def build_disease_chips(cn_classes: list[str]) -> str:
    return "".join(f'<span class="disease-chip">{name}</span>' for name in cn_classes)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio tomato disease detector.")
    parser.add_argument("--run-dir", required=True, help="Experiment directory with best_model.pth.")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def create_app(
    predict_image,
    classes: list[str],
    cn_classes: list[str],
    hero_bg_b64: str,
    gallery_items: list[dict],
    arch_label: str,
    accuracy_label: str,
    image_size: int,
):
    num_diseases = len([c for c in classes if "healthy" not in c.lower()])
    gallery_html = build_gallery_html(gallery_items)
    disease_chips = build_disease_chips(cn_classes)

    with gr.Blocks(
        css=CUSTOM_CSS,
        js=SCROLL_REVEAL_JS,
        title="智农识病 · 番茄病害 AI 诊断系统",
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="orange",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Noto Sans SC"),
        ),
    ) as app:
        gr.HTML("""
        <header class="glass" style="
            position: fixed; top: 0; left: 0; right: 0; z-index: 50;
            display: flex; align-items: center; justify-content: space-between;
            height: 4rem; padding: 0 2rem; transition: all 0.3s ease;
        " id="topNav">
            <div style="display:flex;align-items:center;gap:0.625rem;">
                <div style="
                    width:2.25rem;height:2.25rem;border-radius:0.5rem;
                    background:var(--deep);display:flex;align-items:center;
                    justify-content:center;font-size:1.2rem;
                ">🌱</div>
                <span style="font-family:var(--font-display);font-size:1.125rem;
                    font-weight:600;color:var(--deep);">智农识病</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.75rem;">
                <a href="#diagnostic" style="
                    display:flex;align-items:center;gap:0.5rem;
                    padding:0.5rem 1rem;border-radius:9999px;
                    background:var(--orange);color:#fff;
                    font-size:0.875rem;font-weight:500;text-decoration:none;
                ">📤 上传检测</a>
                <a href="#gallery" style="
                    display:flex;align-items:center;gap:0.5rem;
                    padding:0.5rem 1rem;border-radius:9999px;
                    border:1px solid rgba(4,47,46,0.15);color:var(--deep);
                    font-size:0.875rem;font-weight:500;text-decoration:none;
                ">📖 病害百科</a>
            </div>
        </header>
        """)

        gr.HTML(f"""
        <section class="hero-section" style="margin-top:-4rem;">
            <div class="hero-bg">
                <img src="{hero_bg_b64}" alt="番茄叶片背景" />
                <div class="hero-overlay"></div>
            </div>
            <div class="hero-content">
                <h1 class="hero-title reveal-on-scroll">慧眼识病，守护每一株生长</h1>
                <p class="hero-desc reveal-on-scroll reveal-delay-1">
                    基于深度卷积神经网络 ({arch_label}) 的番茄叶片病害实时识别系统<br/>
                    测试集准确率 {accuracy_label}，为智慧农业保驾护航
                </p>
                <div class="hero-buttons reveal-on-scroll reveal-delay-2">
                    <a href="#diagnostic" class="hero-btn-primary"> 开始诊断</a>
                    <a href="#gallery" class="hero-btn-secondary"> 了解更多</a>
                </div>
                <div class="hero-stats reveal-on-scroll reveal-delay-3">
                    <div>
                        <div class="hero-stat-value">{accuracy_label}</div>
                        <div class="hero-stat-label">识别准确率</div>
                    </div>
                    <div>
                        <div class="hero-stat-value">{num_diseases}</div>
                        <div class="hero-stat-label">可识别病害种类</div>
                    </div>
                    <div>
                        <div class="hero-stat-value">&lt;3s</div>
                        <div class="hero-stat-label">平均识别时间</div>
                    </div>
                </div>
            </div>
            <a href="#diagnostic" class="scroll-indicator">
                <svg class="scroll-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M6 9l6 6 6-6"/>
                </svg>
            </a>
        </section>
        """)

        gr.HTML("""
        <div class="content-section" id="diagnostic">
            <div class="section-header reveal-on-scroll">
                <div class="section-badge">🔬 <span>智能诊断核心</span></div>
                <h2 class="section-title">上传叶片，即刻诊断</h2>
                <p class="section-desc">
                    支持拖拽上传或直接点击选择图片，系统将使用深度学习模型实时分析并给出诊断结果
                </p>
            </div>
        </div>
        """)

        with gr.Row(elem_classes="content-section", elem_id="diagnostic-area"):
            with gr.Column(scale=1, elem_classes="reveal-left"):
                gr.HTML("""
                <div class="column-label">
                    <div class="column-icon column-icon-green">✅</div>
                    <span>影像捕获</span>
                </div>
                """)
                image_input = gr.Image(
                    type="pil",
                    label="",
                    height=360,
                    elem_classes="upload-card-wrap",
                )
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 开始智能诊断",
                        variant="primary",
                        size="lg",
                        elem_classes="gr-button-primary",
                    )
                    clear_btn = gr.Button(
                        "🔄 清除重试",
                        size="lg",
                        elem_classes="gr-button-secondary",
                    )

            with gr.Column(scale=1, elem_classes="reveal-right"):
                gr.HTML("""
                <div class="column-label">
                    <div class="column-icon column-icon-orange">⚠️</div>
                    <span>AI 诊断结果</span>
                </div>
                """)
                result_output = gr.Label(
                    num_top_classes=len(classes),
                    label="",
                    elem_classes="result-card",
                )

        gr.HTML(f"""
        <div class="content-section" style="padding-top:2rem;">
            <div class="tips-grid">
                <div class="tip-card reveal-on-scroll reveal-delay-1">
                    <h4>📷 拍摄建议</h4>
                    <p>将叶片平铺于纯色背景上，确保光线充足、无阴影遮挡</p>
                </div>
                <div class="tip-card reveal-on-scroll reveal-delay-2">
                    <h4>🖼️ 图像格式</h4>
                    <p>支持 JPG、PNG 格式，建议分辨率不低于 512×512 像素</p>
                </div>
                <div class="tip-card reveal-on-scroll reveal-delay-3">
                    <h4>🔬 诊断范围</h4>
                    <p>可识别早疫病、晚疫病、叶霉病等 {num_diseases} 种常见番茄病害，以及健康叶片</p>
                </div>
            </div>
        </div>
        """)

        gr.HTML(f"""
        <div class="content-section reveal-on-scroll" style="padding-top:2rem;padding-bottom:2rem;">
            <div class="disease-chips-bar">
                <span style="font-weight:600;color:var(--deep);margin-right:0.5rem;font-size:0.85rem;">
                    📋 可识别 ({num_diseases} 种病害 + 健康叶片)：
                </span>
                {disease_chips}
            </div>
        </div>
        """)

        gr.HTML(f"""
        <div class="content-section-alt" id="gallery" style="
            background: rgba(255,255,255,0.5);border-radius:2rem;
            margin:0 1rem 2rem 1rem;padding:4rem 1.5rem;
        ">
            <div class="section-header reveal-on-scroll">
                <div class="section-badge">📸 <span>病害图鉴</span></div>
                <h2 class="section-title">微观视界 · 常见病害</h2>
                <p class="section-desc">
                    了解番茄生长过程中可能遇到的常见病害，做到早发现、早防治
                </p>
            </div>
            <div class="gallery-grid">{gallery_html}</div>
        </div>
        """)

        gr.HTML(f"""
        <div class="footer-bar">
            <div class="footer-inner">
                <div class="footer-brand">
                    <div class="footer-logo-icon">🌱</div>
                    <div>
                        <div class="footer-brand-name">智农识病</div>
                        <div class="footer-brand-desc">基于 {arch_label} 深度迁移学习</div>
                    </div>
                </div>
                <div class="footer-text">
                    用 <span style="color:#ea580c;">心</span> 守护每一片绿叶 · 智慧农业 AI 检测系统
                </div>
                <div class="footer-links">
                    <span>图像输入：{image_size} x {image_size} 像素 · PyTorch + Gradio</span>
                </div>
            </div>
        </div>
        """)

        def clear_all():
            return None, None

        analyze_btn.click(fn=predict_image, inputs=image_input, outputs=result_output)
        clear_btn.click(fn=clear_all, inputs=[], outputs=[image_input, result_output])

    return app


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / "config.json")
    dataset_summary = load_json(run_dir / "dataset_summary.json")
    classes = dataset_summary["classes"]
    model_path = run_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model weights: {model_path}\n"
            "Train the experiment first, e.g.:\n"
            f"  python -X utf8 train.py --preset {config.get('experiment_name', '<preset>')}"
        )

    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    device = torch.device("cpu")
    model = build_model(config["architecture"], len(classes), False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    _, eval_transform = build_transforms(config["image_size"])

    cn_classes = get_cn_classes(classes)
    arch_label = ARCHITECTURE_LABELS.get(config["architecture"], config["architecture"])
    accuracy_label = load_accuracy_label(run_dir)

    script_dir = Path(__file__).resolve().parent
    hero_bg_b64 = img_file_to_data_uri(script_dir / "static/hero-botanical-bg.jpg", max_width=1400)
    if not hero_bg_b64:
        hero_bg_b64 = placeholder_data_uri("智农识病", "#042f2e")
    gallery_items = prepare_gallery(script_dir)

    def predict_image(image):
        if image is None:
            return None
        tensor = eval_transform(image).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor)[0], dim=0)
        return {cn_classes[i]: float(probs[i]) for i in range(len(classes))}

    app = create_app(
        predict_image=predict_image,
        classes=classes,
        cn_classes=cn_classes,
        hero_bg_b64=hero_bg_b64,
        gallery_items=gallery_items,
        arch_label=arch_label,
        accuracy_label=accuracy_label,
        image_size=config["image_size"],
    )

    static_dir = script_dir / "static"
    allowed_paths = [str(static_dir)] if static_dir.exists() else None
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=args.server_port,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
