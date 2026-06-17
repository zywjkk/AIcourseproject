# TomatoAI Web 前端 — AI 交接文档

> 写给下一个接手优化的 AI / 开发者。阅读时间约 10 分钟。

---

## 1. 项目概览

**番茄病害智能识别系统** — 浙江大学人工智能基础(A) 课程大作业。

```
项目根目录: D:\Desktop\学业\大一下\人基A\AIcourseproject-main
Web 目录:   web/
  ├── app.py                 # FastAPI 后端（原 server.py，已被用户重命名）
  └── static/
      ├── index.html          # 主页面（单页 SPA，4 个 section 面板切换）
      ├── style.css           # 全局样式（DM Sans + Inter + Noto Sans SC 字体）
      ├── chart.js            # Chart.js 4.x + 自定义 3D 长方体柱状图插件
      └── app.js              # 前端主控逻辑（导航、拖拽上传、API 调用、DOM 渲染）
```

**启动命令**（在项目根目录执行）：
```powershell
python web/app.py --port 8000 --run-dir outputs/experiments/resnet18_transfer_fc_then_layer4
```

浏览器访问 `http://127.0.0.1:8000`。

---

## 2. 后端 `web/app.py`（344 行）

### 核心职责
- 加载 PyTorch 模型（`best_model.pth`）+ 配置（`config.json`）+ 类别列表（`dataset_summary.json`）
- 提供 3 个路由

### 路由表

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回 `static/index.html` |
| `/static/*` | GET | 静态文件挂载（CSS/JS） |
| `/exp-img/{filename}` | GET | 从实验目录（`--run-dir`）直接提供 PNG 图片，用于模型页展示混淆矩阵、Grad-CAM 等 |
| `/api/info` | GET | 返回 JSON：模型架构名、类别列表、准确率 |
| `/api/predict` | POST | 接收 `multipart/form-data`（字段名 `file`），返回预测结果的 JSON |

### `/api/predict` 返回格式
```json
{
  "probabilities": [0.966, 0.012, ...],   // 10 个浮点数，顺序与 classes 一致
  "predicted_class": "Tomato___healthy",
  "predicted_cn": "健康叶片",
  "confidence": 0.966,
  "disease_info": {
    "cn": "健康叶片",
    "desc": "叶片生长正常...",
    "treatment": ["合理施肥...", ...]
  },
  "cn_classes": ["细菌性斑点病", "早疫病", ...]
}
```

### 关键设计
- `DISEASE_INFO` 字典（约 120 行）内嵌了 10 种病害的中文名、症状描述、5 条防治建议——这些数据同时被 API 返回给前端渲染，不在 HTML 里硬编码。
- `load_tomato_model()` 函数从 `config.json` 读取架构名动态构建模型（支持 resnet18/resnet50/mobilenet_v3_small），仅依赖 `torchvision.models`。
- 模型权重用 `weights_only=True` 加载，安全性更好。
- 图片输入统一 `RGB` 转换 + ImageNet 归一化。

---

## 3. 前端架构（SPA 单页应用）

### 3.1 `index.html`（291 行）

单文件包含 4 个 `<section>` 面板，通过左侧固定侧边栏的按钮切换（`data-page` 属性驱动）。

| 面板 | id | 内容 |
|------|----|------|
| 使用模型 | `page-predict` | 图片拖拽上传区 + 诊断按钮 + 3D 柱状图 + 病害信息卡 |
| 项目介绍 | `page-about` | 背景、目标、技术路线、数据集 4 张卡片 + GitHub 开源链接 |
| 模型架构 | `page-model` | ResNet-18 参数表 + 测试集性能表 + 各类别分类报告表 + 混淆矩阵图 + Grad-CAM 图 + Loss/Acc 训练曲线 |
| 团队成员 | `page-contributors` | 5 人分工表（组长郑喻文 + 方鑫/王安卓/李军/王骏扬） |

### 3.2 侧边栏设计

- 4 个导航按钮，每个有不同的主题色：
  - 使用模型 → 锈红 `#e85d4a`
  - 项目介绍 → 蓝色 `#58a6ff`
  - 模型架构 → 绿色 `#3fb950`
  - 团队成员 → 金色 `#f5a623`
- 激活态左侧有 4px 宽竖条标记
- 底部固定显示作者信息
- 侧边栏宽 260px，响应式折叠到 60px

### 3.3 字体方案
```
Headings: "DM Sans" → "Noto Sans SC" → system sans-serif
Body:     "Inter" → "Noto Sans SC" → system sans-serif
```
通过 Google Fonts CDN 加载（在 `<head>` 中 `preconnect` + 单次 `link` 请求）。

---

## 4. 核心 JS 模块

### 4.1 `app.js`（153 行）— 主控逻辑

**全局变量**：`selectedFile`（当前选中的图片 File 对象）

**关键函数**：
- `handleFile(file)` — FileReader 读取预览，启用诊断按钮，调用 `resetResults()`
- `renderResults(data)` — 核心渲染：
  1. `renderProbChart('probChart', cn_classes, probabilities)` 画 3D 柱状图
  2. 填充预测 badge（中文名 + 置信度百分比）
  3. 填充病害信息卡（描述 + 防治建议 `<li>` 列表）

**导航**：遍历 `navItems`，点击时切换 `.active` 类，跳转到模型页时延迟 150ms 调用 `createTrainingCharts()`（等 DOM 渲染完成）。

**错误处理**：`/api/predict` 失败时 `alert()` 弹窗显示错误信息。

### 4.2 `chart.js`（224 行）— 图表模块

#### 3D 长方体柱状图插件 `cuboid3dPlugin`

这是整个项目最复杂的自定义功能。Chart.js v4 没有原生 3D 支持，所以用 `afterDatasetDraw` 钩子在每个 bar 渲染后手绘右侧面和顶面平行四边形。

**几何原理**：
```
        top-left ──── top-right          ← 顶面（亮色 = 原色 × 1.35）
             ╲          ╲
              ╲          ╲
           front-left  front-right       ← 正面（Chart.js 原生渲染）
               │          │
               │          │
            bottom-left bottom-right     ← 右侧面（暗色 = 原色 × 0.65）
```

**三面配色规则**（按光照法则）：
- **正面**：Chart.js 使用 `backgroundColor` 数组中的原始颜色
- **右侧面**：`rgba(r×0.65, g×0.65, b×0.65, 1)` — 原色变暗 35%
- **顶面**：`rgba(r×1.35, g×1.35, b×1.35, 1)` — 原色提亮 35%（上限 255）
- `hexToRgb()` 辅助函数解析 `#rrggbb` 格式

**实现细节**：
- 深度投影：水平 12px，垂直 8px（模拟等距投影）
- 顶面有 0.6px 描边增加棱角感
- 仅在有数据时才绘制（`meta.data.length > 0` 守卫）

#### `renderProbChart(canvasId, labels, values)`

- 每次调用先 `destroy()` 旧实例（防止 "Canvas already in use" 错误）
- **只显示 Top-5 最高概率类别**（排序 + 切片）
- **FLOOR 机制**：所有值 clamp 到 `≥0.025`，确保近零概率的 bar 仍有可视高度
- **Tooltip 显示原始值**：通过 `dataset._original` 数组存储未经 clamp 的真实概率
- 动画 `duration: 1500ms, easing: easeOutQuart`
- `borderRadius: 0`（锐利棱角），`barPercentage: 0.68`

#### `createTrainingCharts()`

- 在切换到模型页时调用
- 绘制两个 `<canvas>`：Loss 曲线 + Accuracy 曲线
- 数据硬编码为 30 epoch 的真实训练历史（来自 `history.json`）
- 每次调用先 destroy 旧图表

---

## 5. CSS 设计系统 `style.css`（297 行）

### 设计语言
- **灵感来源**：GitHub 干净现代的 UI 风格
- **配色**：深灰侧边栏 `#1c2128` + 浅灰主背景 `#f6f8fa` + 白卡片 `#ffffff`
- **浅色卡片**：模型页和介绍页使用 `--bg-tinted: #f3f5f7`（而非纯白）形成视觉分层
- **防治建议**：浅绿底色 `#e6ffec` + 浅绿边框 `#bef5cb`，类似 GitHub 的 note callout
- **圆角**：8px（小）/ 12px（大）
- **阴影**：三层级（sm/md/lg），极浅

### 关键 CSS 变量
```css
--sidebar-bg: #1c2128;    /* 不是纯黑，是深灰 */
--bg-tinted: #f3f5f7;     /* 浅卡片底色 */
--green-bg: #e6ffec;      /* 防治建议底色 */
```

### 响应式
- `max-width: 900px` 时预测页双栏变单栏
- `max-width: 768px` 时侧边栏折叠为 60px 仅显示图标

---

## 6. 已修复的 Bug（重要！不要重蹈覆辙）

| # | 症状 | 根因 | 修复方式 |
|---|------|------|----------|
| 1 | `Canvas is already in use` | 同一 canvas 上重复 `new Chart()` 未 destroy | `renderProbChart` 和 `createTrainingCharts` 开头先 `destroy()` + `null` |
| 2 | `Cannot read properties of undefined (reading 'data')` | 先用空数据建 chart，再 `update()` 填真数据，Chart.js 内部 `_metasets` 未初始化就触发了 `afterDatasetDraw` | 改为 `renderProbChart` 一次性用真数据创建，不再分两步 |
| 3 | `experiment_summary.csv` 重复行 | `train.py` 的 `_append_summary` 用追加模式 | 改成 upsert：先读取所有行 → 删同名旧行 → 写回全部 |
| 4 | `window._probChartInstance` vs `let probChart` 不同变量 | 模块作用域混乱 | 统一用 `let probChart` 模块变量，不用 `window.*` |
| 5 | 柱状图顶面不可见 | 透明度太低 `rgba(255,255,255,0.30)` | 改为根据 bar 原色提亮 35% 的实色 |

---

## 7. 与后端模型的耦合点

前端不直接操作 PyTorch。图片通过 `fetch('/api/predict', {method:'POST', body: FormData})` 发给后端。

**类别顺序敏感**：`cn_classes` 数组的顺序由 `server.py` 的 `CN_CLASSES` 常量硬编码，必须与模型输出 logits 的顺序一致。这个顺序来源于 `torchvision.datasets.ImageFolder` 按字母排序的 `classes`。

**实验图片路径**：模型页的 `<img src="/exp-img/...">` 走 `/exp-img/{filename}` 路由，后端直接从 `--run-dir` 参数指定的目录读取文件。这要求该目录下有 `confusion_matrix.png`、`grad_cam.png` 等文件。

---

## 8. 其他今天修改的文件

- **`train.py`**（121-159 行）：`_append_summary` 函数改为 upsert 逻辑，消除 CSV 重复行
- **项目根目录 `app.py`** 已被用户删除（原 Gradio 前端，功能已由 `web/` 替代）
- **`.gitignore`** 无变动

---

## 9. 已知可改进点（未实现）

1. **柱状图近零 bar 的 FLOOR 值**：目前 0.025 是经验值，如果所有类别概率都很低（如错误图片），全部被 clamp 到 2.5% 会让图表看起来不合理。可考虑动态判断：如果最高概率 < 0.5，则降低 FLOOR。
2. **模型页的图片加载**：`confusion_matrix.png` 和 `grad_cam.png` 较大（~200KB），已有 `loading="lazy"` 但可进一步加 placeholder skeleton。
3. **训练曲线数据**：目前硬编码在 `chart.js` 的 `HIST` 对象中。可以从 `/api/info` 扩展一个 `/api/history` 端点返回 `history.json`。
4. **多模型切换**：目前只加载一个模型（`--run-dir` 参数）。前端模型介绍页展示了 ResNet-18/50/MobileNet-V3 的对比数据，但诊断只用一个模型。
5. **DISEASE_INFO 重复**：防治建议在后端 `server.py` 和前端渲染逻辑中各出现一次（后端返回、前端渲染）。如果修改病害信息需要改后端。
6. **错误图片处理**：`/api/predict` 收到的如果不是番茄叶片（如风景照），模型仍会给出最高概率的类别。前端未做"非叶片图片"的检测。

---

## 10. 文件清单（当前最终状态）

```
web/
├── app.py              344 行  FastAPI 后端
└── static/
    ├── index.html       291 行  主页面
    ├── style.css        297 行  全局样式
    ├── chart.js         224 行  3D 柱状图 + 训练曲线
    └── app.js           153 行  前端主控
─────────────────────────────────
总计                   1,309 行
```
