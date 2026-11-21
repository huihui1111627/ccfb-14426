package main

const indexHTML = `<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>流水线并行训练调度器</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px; color: #222; }
  h1 { font-size: 20px; } h2 { font-size: 16px; margin: 18px 0 6px; }
  form { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: end; background: #f6f7f9; padding: 12px 16px; border-radius: 8px; }
  label { display: flex; flex-direction: column; font-size: 12px; color: #555; }
  input { width: 90px; padding: 4px 6px; font-size: 13px; }
  input.wide { width: 150px; }
  button { padding: 6px 18px; font-size: 14px; background: #2563eb; color: #fff; border: 0; border-radius: 6px; cursor: pointer; }
  table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
  th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: right; }
  th { background: #f0f1f3; } td:first-child, th:first-child { text-align: left; }
  canvas { border: 1px solid #e2e2e2; border-radius: 6px; margin: 6px 0; background: #fff; }
  .bar { height: 14px; border-radius: 3px; display: inline-block; vertical-align: middle; }
  .note { color: #777; font-size: 12px; }
  .legend span { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin: 0 4px 0 12px; vertical-align: middle; }
</style>
</head>
<body>
<h1>流水线并行训练调度器 <span class="note">古桥裂缝/风化识别模型 · 每层网络切分到多张卡</span></h1>
<form id="cfg">
  <label>层数 L<input name="layers" value="12"></label>
  <label>卡数 p<input name="devices" value="4"></label>
  <label>微批数 m<input name="microbatches" value="8"></label>
  <label>虚拟段数 v<input name="vchunks" value="2"></label>
  <label>各卡算力(逗号分隔)<input class="wide" name="speeds" value="1,0.4,1,1"></label>
  <label>传输时延<input name="latency" value="0.15"></label>
  <label>前向单价<input name="fwdcost" value="1"></label>
  <label>反向单价<input name="bwdcost" value="2"></label>
  <button type="submit">运行</button>
</form>
<div class="legend" style="margin:8px 0">
  <span style="background:#3b82f6"></span>前向 F
  <span style="background:#f59e0b"></span>反向 B
  <span style="background:#facc15"></span>卡间传输
  <span style="background:#e5e7eb"></span>气泡(空闲)
</div>
<div id="summary"></div>
<div id="charts"></div>
<script>
const $ = s => document.querySelector(s);
const pct = x => (x*100).toFixed(1) + "%";
const f2 = x => x.toFixed(2);

async function run() {
  const q = new URLSearchParams(new FormData($("#cfg"))).toString();
  const data = await (await fetch("/api/run?" + q)).json();
  renderSummary(data);
  renderCharts(data);
}

function renderSummary(data) {
  let h = "<h2>各调度总耗时对比（当前配置 vs 理想同构卡）</h2><table><tr>" +
    "<th>调度</th><th>总耗时(当前)</th><th>总耗时(理想)</th><th>气泡占比(实测)</th><th>气泡占比(理论)</th>" +
    "<th>峰值激活(实测/理论, 按卡)</th><th></th></tr>";
  const maxMk = Math.max(...data.schedules.map(s => s.makespan));
  data.schedules.forEach((s, i) => {
    const peak = s.peak.map((p, d) => p + "/" + s.peakTheory[d]).join(" ");
    const w = (s.makespan / maxMk * 160).toFixed(0);
    h += "<tr><td>" + s.name + "</td><td>" + f2(s.makespan) + "</td><td>" + f2(data.ideal[i].makespan) +
      "</td><td>" + pct(s.bubble) + "</td><td>" + pct(s.bubbleTheory) + "</td><td>" + peak +
      '</td><td><span class="bar" style="width:' + w + 'px;background:#93c5fd"></span></td></tr>';
  });
  $("#summary").innerHTML = h + "</table>" +
    '<p class="note">理论值按公式：GPipe/1F1B 气泡=(p-1)/(m+p-1)，交错=(p-1)/(m·v+p-1)（设备瓶颈区间）；' +
    '峰值激活：GPipe=m，1F1B 第 d 卡=min(p-d,m)，交错=m·v（段激活单位）。</p>';
}

function renderCharts(data) {
  const div = $("#charts");
  div.innerHTML = "";
  data.schedules.forEach(s => {
    const h2 = document.createElement("h2");
    h2.textContent = s.name + " — 时序图（总耗时 " + f2(s.makespan) + "，气泡 " + pct(s.bubble) + "）";
    div.appendChild(h2);
    const cv = document.createElement("canvas");
    const rowH = 30, marginL = 56, marginR = 12, W = 1100;
    cv.width = W; cv.height = marginL * 0 + data.p * rowH + 24;
    cv.style.width = "100%"; cv.style.maxWidth = W + "px";
    div.appendChild(cv);
    drawGantt(cv, s, data.p, data.s, rowH, marginL, marginR);
  });
}

function drawGantt(cv, s, p, nChunks, rowH, mL, mR) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const scale = (W - mL - mR) / s.makespan;
  ctx.font = "10px sans-serif";
  // rows + idle background
  for (let d = 0; d < p; d++) {
    const y = d * rowH + 4;
    ctx.fillStyle = "#e5e7eb";
    ctx.fillRect(mL, y, s.makespan * scale, rowH - 8);
    ctx.fillStyle = "#333";
    ctx.fillText("卡 " + d, 12, y + rowH / 2 + 3);
  }
  // transfers (thin yellow strip at top of target row)
  for (const t of s.transfers) {
    ctx.fillStyle = "#facc15";
    ctx.fillRect(mL + t.start * scale, t.to * rowH + 4, Math.max(1, (t.end - t.start) * scale), 4);
  }
  // tasks
  for (const e of s.events) {
    const y = e.dev * rowH + 4, x = mL + e.start * scale, w = Math.max(1, (e.end - e.start) * scale);
    ctx.fillStyle = e.kind === "F" ? "#3b82f6" : "#f59e0b";
    ctx.fillRect(x, y, w, rowH - 8);
    ctx.strokeStyle = "rgba(0,0,0,.25)";
    ctx.strokeRect(x + .5, y + .5, w - 1, rowH - 9);
    if (w > 26) {
      ctx.fillStyle = "#fff";
      const label = e.kind + e.mb + (nChunks > p ? "·" + e.chunk : "");
      ctx.fillText(label, x + 3, y + rowH / 2 + 3);
    }
  }
  // time axis
  ctx.fillStyle = "#666";
  const step = niceStep(s.makespan / 8);
  for (let t = 0; t <= s.makespan + 1e-9; t += step) {
    const x = mL + t * scale;
    ctx.fillRect(x, H - 14, 1, 4);
    ctx.fillText(f2(t), x - 8, H - 3);
  }
}

function niceStep(x) {
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  for (const m of [1, 2, 5, 10]) if (m * p >= x) return m * p;
  return 10 * p;
}

$("#cfg").addEventListener("submit", e => { e.preventDefault(); run(); });
run();
</script>
</body>
</html>
`
