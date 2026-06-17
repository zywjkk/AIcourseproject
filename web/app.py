"""
Tomato Disease Recognition — FastAPI backend.

Launch:
    cd web
    python server.py --port 8000 --run-dir ../outputs/experiments/resnet18_transfer_fc_then_layer4
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# FastAPI with fallback to simple HTTP server
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_tomato_model(run_dir: str):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text("utf-8"))
    summary = json.loads((run_dir / "dataset_summary.json").read_text("utf-8"))
    classes = summary["classes"]
    image_size = config["image_size"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build model matching src/models.py
    arch = config["architecture"]
    import torch.nn as nn

    if arch == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if config["pretrained"] else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif arch == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V2 if config["pretrained"] else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif arch == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = (
            MobileNet_V3_Small_Weights.IMAGENET1K_V1 if config["pretrained"] else None
        )
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, len(classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.load_state_dict(
        torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, eval_transform, classes, device, config


# ---------------------------------------------------------------------------
# Disease information (from agricultural extension sources)
# ---------------------------------------------------------------------------
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "cn": "细菌性斑点病",
        "desc": "由黄单胞杆菌（Xanthomonas campestris pv. vesicatoria）引起。叶片出现水渍状小斑点，后变黑褐色、周围黄化，严重时叶片枯死。高温高湿环境下易爆发。",
        "treatment": [
            "选用无病种子，播种前用 50℃ 温水浸种 25 分钟",
            "实行与非茄科作物 2-3 年轮作",
            "发病初期喷洒氢氧化铜（可杀得）500 倍液，每 7-10 天一次",
            "加强田间通风，避免植株表面长时间结露",
            "及时清除病残体，收获后深翻土壤",
        ],
    },
    "Tomato___Early_blight": {
        "cn": "早疫病",
        "desc": "由茄链格孢菌（Alternaria solani）引起。叶片出现同心轮纹状褐色病斑，周围有黄色晕圈。多从下部老叶开始发病，逐渐向上蔓延。",
        "treatment": [
            "选用抗病品种，合理密植保证通风",
            "覆膜栽培，减少土壤中的病原菌溅到叶片上",
            "发病前喷施代森锰锌（大生）600 倍液预防",
            "发病后用嘧菌酯（阿米西达）1500 倍液或苯醚甲环唑（世高）2000 倍液喷雾",
            "及时摘除下部老叶、病叶并带出田外销毁",
        ],
    },
    "Tomato___Late_blight": {
        "cn": "晚疫病",
        "desc": "由致病疫霉菌（Phytophthora infestans）引起，是番茄最具毁灭性的病害。叶片出现水渍状暗绿色斑点，湿度大时叶背有白色霉层，病斑迅速扩展导致整株枯死。",
        "treatment": [
            "天气预报有连续阴雨时，提前喷施保护性杀菌剂如霜脲·锰锌（克露）600 倍液",
            "发病初期立即喷施氟菌·霜霉威（银法利）600 倍液或烯酰吗啉 1500 倍液",
            "控制田间湿度，采用膜下滴灌，避免大水漫灌",
            "发现中心病株立即拔除深埋，周围植株重点喷药保护",
            "收获后彻底清除田间病残体",
        ],
    },
    "Tomato___Leaf_Mold": {
        "cn": "叶霉病",
        "desc": "由褐孢霉菌（Passalora fulva，旧称 Fulvia fulva）引起。叶面出现淡黄色褪绿斑，叶背密生灰褐色绒状霉层。主要危害保护地（大棚）番茄。",
        "treatment": [
            "选用抗叶霉病品种（如含有 Cf-9 抗性基因的品种）",
            "加强通风换气，大棚内相对湿度控制在 80% 以下",
            "发病初期喷施嘧菌酯（阿米西达）1500 倍液或甲基硫菌灵（甲基托布津）800 倍液",
            "用 45% 百菌清烟剂熏棚，每亩 250g，傍晚密闭棚室熏烟",
            "与非茄科作物轮作 2-3 年",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "cn": "斑枯病",
        "desc": "由番茄壳针孢菌（Septoria lycopersici）引起。叶片出现圆形水渍状小斑点，边缘深褐色、中央灰白色，后期病斑上可见黑色小颗粒（分生孢子器）。",
        "treatment": [
            "实行 2-3 年轮作，避免连作",
            "合理密植，及时整枝打杈，改善通风透光",
            "发病前喷施代森锰锌（大生）600 倍液保护",
            "发病后喷施苯醚甲环唑（世高）2000 倍液或戊唑醇 3000 倍液",
            "清除田间病残体，集中烧毁或深埋",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cn": "二斑叶螨（红蜘蛛）",
        "desc": "由二斑叶螨（Tetranychus urticae）危害。螨虫在叶背刺吸汁液，叶片出现黄白色小斑点，严重时叶片枯黄脱落，叶背可见细丝网。高温干燥条件下繁殖极快。",
        "treatment": [
            "保持田间湿度，干旱季节适当喷水，抑制螨虫繁殖",
            "释放捕食螨如智利小植绥螨（Phytoseiulus persimilis）进行生物防治",
            "药剂防治：阿维菌素 1500 倍液或联苯肼酯（爱卡螨）2000 倍液喷雾",
            "重点喷叶背，每 7 天一次，连续 2-3 次，注意轮换用药避免抗性",
            "清除田边杂草，减少越冬虫源",
        ],
    },
    "Tomato___Target_Spot": {
        "cn": "靶斑病",
        "desc": "由多主棒孢霉菌（Corynespora cassiicola）引起。病斑圆形至不规则形，中央灰白色、边缘褐色，呈明显同心轮纹状似靶标，严重时叶片穿孔。",
        "treatment": [
            "选用无病种子，进行种子消毒处理",
            "合理施肥，增施有机肥和磷钾肥，增强植株抗性",
            "发病初期喷洒嘧菌酯（阿米西达）1500 倍液或咪鲜胺 1000 倍液",
            "与百菌清、代森锰锌等保护性杀菌剂交替使用",
            "注意田间排水，降低湿度",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cn": "黄化曲叶病毒病",
        "desc": "由番茄黄化曲叶病毒（TYLCV）引起，通过烟粉虱传播。植株矮缩、顶部叶片黄化卷曲变小，花器发育不良，果实少而小。一旦感染无法治愈。",
        "treatment": [
            "选用抗 TYLCV 品种（这是最有效的措施）",
            "苗期和生长期覆盖 60 目防虫网，隔绝烟粉虱",
            "悬挂黄板监测和诱杀烟粉虱成虫",
            "药剂防虫：吡虫啉 1500 倍液或噻虫嗪（阿克泰）3000 倍液灌根+喷雾",
            "及时拔除并销毁病株，减少病毒源",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "cn": "花叶病毒病",
        "desc": "由烟草花叶病毒（TMV）或番茄花叶病毒（ToMV）引起。叶片呈深浅绿色相间的花叶斑驳，严重时叶片变细呈蕨叶状。病毒通过汁液接触传播（农事操作、修剪等）。",
        "treatment": [
            "选用抗 TMV/ToMV 品种（含 Tm-2² 抗性基因）",
            "播种前种子用 10% 磷酸三钠溶液浸泡 20 分钟消毒",
            "农事操作前用肥皂水洗手，工具用 3% 磷酸三钠溶液消毒",
            "及时清除田间病株，操作时先管健康株再管病株",
            "苗期喷施盐酸吗啉胍（病毒 A）500 倍液，提高植株抗性",
        ],
    },
    "Tomato___healthy": {
        "cn": "健康叶片",
        "desc": "叶片生长正常，无病斑、无虫害迹象。保持健康的栽培管理措施，预防胜于治疗。",
        "treatment": [
            "合理施肥，N:P:K 比例约 1:0.5:1.2，增施腐熟有机肥",
            "保持适宜土壤湿度，采用滴灌或膜下灌溉",
            "定期巡查田间，做到早发现、早预防",
            "实行轮作倒茬，避免连作障碍",
            "合理整枝打杈，保持田间通风透光",
        ],
    },
}

# Chinese display names for bar chart
CN_CLASSES = [
    "细菌性斑点病",
    "早疫病",
    "晚疫病",
    "叶霉病",
    "斑枯病",
    "二斑叶螨",
    "靶斑病",
    "黄化曲叶病毒病",
    "花叶病毒病",
    "健康叶片",
]


def get_disease_info(class_name: str) -> dict:
    info = DISEASE_INFO.get(class_name)
    if info is None:
        return {
            "cn": class_name.replace("Tomato___", "").replace("_", " "),
            "desc": "暂无详细信息。",
            "treatment": ["请咨询当地农业技术推广部门。"],
        }
    return info


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
def create_app(run_dir: str) -> FastAPI:
    model, transform, classes, device, config = load_tomato_model(run_dir)

    app = FastAPI(
        title="Tomato Disease Recognition API",
        version="1.0",
        docs_url=None,  # hide docs in production
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/info")
    async def api_info():
        return JSONResponse(
            {
                "classes": classes,
                "cn_classes": CN_CLASSES,
                "architecture": config["architecture"],
                "accuracy": 0.974,
                "parameters": "11.18M",
            }
        )

    @app.post("/api/predict")
    async def api_predict(file: UploadFile = File(...)):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits[0], dim=0)

        probabilities = probs.cpu().numpy().tolist()
        max_idx = int(probs.argmax().item())
        predicted_class = classes[max_idx]
        info = get_disease_info(predicted_class)

        return JSONResponse(
            {
                "probabilities": [round(p, 6) for p in probabilities],
                "predicted_class": predicted_class,
                "predicted_cn": info["cn"],
                "confidence": round(probabilities[max_idx], 4),
                "disease_info": info,
                "cn_classes": CN_CLASSES,
            }
        )

    # Serve experiment images
    exp_dir = Path(run_dir)

    @app.get("/exp-img/{filename}")
    async def exp_image(filename: str):
        file_path = exp_dir / filename
        if not file_path.is_file():
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse("Not Found", status_code=404)
        return FileResponse(file_path)

    # Serve static files
    static_dir = Path(__file__).parent / "static"

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    # Mount static files (CSS, JS)
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tomato Disease Recognition Server")
    parser.add_argument(
        "--run-dir",
        default="../outputs/experiments/resnet18_transfer_fc_then_layer4",
        help="Experiment directory with best_model.pth",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    args = parser.parse_args()

    if not HAS_FASTAPI:
        print("Error: fastapi and uvicorn are required.")
        print("  pip install fastapi uvicorn python-multipart")
        return

    run_dir = str(Path(args.run_dir).resolve())
    app = create_app(run_dir)
    print(f"\n  Server: http://{args.host}:{args.port}")
    print(f"  Model:  {run_dir}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
