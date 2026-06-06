export const C = {
  bg: "#FAFAF7",
  ink: "#17201A",
  muted: "#647067",
  faint: "#DCE2DA",
  pale: "#EEF3EC",
  green: "#2F6B4F",
  green2: "#5F8D6A",
  gold: "#B58A25",
  red: "#A44E42",
  white: "#FFFFFF",
};

export function base(slide, ctx, n, section = "番茄病害识别检测系统") {
  ctx.addShape(slide, { left: 0, top: 0, width: ctx.W, height: ctx.H, fill: C.bg });
  ctx.addShape(slide, { left: 48, top: 50, width: 1184, height: 1, fill: C.faint });
  ctx.addText(slide, {
    text: section,
    left: 50,
    top: 24,
    width: 620,
    height: 22,
    fontSize: 12,
    color: C.muted,
    typeface: "Microsoft YaHei",
  });
  ctx.addText(slide, {
    text: String(n).padStart(2, "0"),
    left: 1186,
    top: 22,
    width: 46,
    height: 24,
    fontSize: 13,
    color: C.green,
    bold: true,
    align: "right",
    typeface: "Aptos",
  });
}

export function title(slide, ctx, text, sub = "") {
  ctx.addText(slide, {
    text,
    left: 50,
    top: 70,
    width: 720,
    height: 52,
    fontSize: 30,
    color: C.ink,
    bold: true,
    typeface: "Microsoft YaHei",
  });
  if (sub) {
    ctx.addText(slide, {
      text: sub,
      left: 52,
      top: 122,
      width: 860,
      height: 30,
      fontSize: 13,
      color: C.muted,
      typeface: "Microsoft YaHei",
    });
  }
}

export function label(slide, ctx, text, x, y, w = 180, color = C.green) {
  ctx.addText(slide, {
    text,
    left: x,
    top: y,
    width: w,
    height: 20,
    fontSize: 11,
    color,
    bold: true,
    typeface: "Microsoft YaHei",
  });
}

export function metric(slide, ctx, value, name, x, y, w = 168) {
  ctx.addShape(slide, {
    left: x,
    top: y,
    width: w,
    height: 78,
    fill: C.white,
    line: { fill: C.faint, width: 1, style: "solid" },
  });
  ctx.addText(slide, {
    text: value,
    left: x + 14,
    top: y + 12,
    width: w - 28,
    height: 34,
    fontSize: 25,
    color: C.green,
    bold: true,
    typeface: "Aptos Display",
  });
  ctx.addText(slide, {
    text: name,
    left: x + 14,
    top: y + 48,
    width: w - 28,
    height: 18,
    fontSize: 10,
    color: C.muted,
    typeface: "Microsoft YaHei",
  });
}

export function bulletList(slide, ctx, items, x, y, w, gap = 30, size = 15) {
  items.forEach((item, i) => {
    const yy = y + i * gap;
    ctx.addShape(slide, { left: x, top: yy + 7, width: 5, height: 5, fill: item.color || C.green });
    ctx.addText(slide, {
      text: item.text || item,
      left: x + 18,
      top: yy,
      width: w - 18,
      height: gap,
      fontSize: size,
      color: item.muted ? C.muted : C.ink,
      bold: Boolean(item.bold),
      typeface: "Microsoft YaHei",
    });
  });
}

export function smallTable(slide, ctx, columns, rows, x, y, widths, rowH = 32) {
  const totalW = widths.reduce((a, b) => a + b, 0);
  ctx.addShape(slide, { left: x, top: y, width: totalW, height: rowH, fill: C.green });
  let xx = x;
  columns.forEach((c, i) => {
    ctx.addText(slide, {
      text: c,
      left: xx + 8,
      top: y + 8,
      width: widths[i] - 16,
      height: 18,
      fontSize: 10,
      color: C.white,
      bold: true,
      typeface: "Microsoft YaHei",
    });
    xx += widths[i];
  });
  rows.forEach((r, ri) => {
    const yy = y + rowH * (ri + 1);
    ctx.addShape(slide, {
      left: x,
      top: yy,
      width: totalW,
      height: rowH,
      fill: ri % 2 === 0 ? C.white : C.pale,
      line: { fill: C.faint, width: 0.5, style: "solid" },
    });
    let cx = x;
    r.forEach((cell, ci) => {
      ctx.addText(slide, {
        text: String(cell),
        left: cx + 8,
        top: yy + 8,
        width: widths[ci] - 16,
        height: 18,
        fontSize: 10,
        color: ci === 0 ? C.ink : C.muted,
        bold: ci === 0,
        typeface: "Microsoft YaHei",
      });
      cx += widths[ci];
    });
  });
}

export function bar(slide, ctx, labelText, value, max, x, y, w, color = C.green) {
  const bw = Math.max(2, (value / max) * w);
  ctx.addText(slide, {
    text: labelText,
    left: x,
    top: y - 4,
    width: 230,
    height: 18,
    fontSize: 10,
    color: C.ink,
    typeface: "Microsoft YaHei",
  });
  ctx.addShape(slide, { left: x + 236, top: y, width: w, height: 10, fill: C.faint });
  ctx.addShape(slide, { left: x + 236, top: y, width: bw, height: 10, fill: color });
  ctx.addText(slide, {
    text: `${(value * 100).toFixed(1)}%`,
    left: x + 244 + w,
    top: y - 4,
    width: 50,
    height: 18,
    fontSize: 10,
    color,
    bold: true,
    typeface: "Aptos",
  });
}

export function miniNode(slide, ctx, text, x, y, w = 150, h = 42, fill = C.white) {
  ctx.addShape(slide, {
    left: x,
    top: y,
    width: w,
    height: h,
    fill,
    line: { fill: C.green2, width: 1, style: "solid" },
  });
  ctx.addText(slide, {
    text,
    left: x + 8,
    top: y + 10,
    width: w - 16,
    height: h - 12,
    fontSize: 11,
    color: C.ink,
    align: "center",
    typeface: "Microsoft YaHei",
  });
}
