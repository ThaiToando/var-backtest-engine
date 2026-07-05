"""
Generates a self-contained, animated HTML page visualizing 300 simulated
60-day forward portfolio paths from the Monte Carlo (Student-t) model.
Paths breaching the 99% horizon VaR threshold are highlighted, so viewing
this page IS a live demonstration of the fat-tail risk this project is about.
"""
import os
import json
import numpy as np

from src.data_loader import load_config, load_prices
from src.returns import simple_returns, compute_weights
from src.var_montecarlo import simulate_multiday_paths

OUTPUT_PATH = os.path.join("outputs", "monte_carlo_paths.html")

NUM_PATHS = 300
HORIZON_DAYS = 60
CONFIDENCE = 0.99


def build_html(paths: np.ndarray, var_threshold: float, es: float, distribution: str, df: int) -> str:
    paths_pct = (paths * 100).round(3).tolist()  # convert to % for readability in JS
    var_pct = round(var_threshold * 100, 3)
    es_pct = round(es * 100, 3)

    data_json = json.dumps(paths_pct)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Monte Carlo Simulated Portfolio Paths</title>
<style>
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #FAFAFA;
    color: #2C3E50;
    margin: 0;
    padding: 40px;
  }}
  .container {{ max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  p.subtitle {{ color: #666; margin-top: 0; margin-bottom: 24px; font-size: 14px; }}
  #chart-wrap {{
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  }}
  canvas {{ display: block; width: 100%; height: 460px; }}
  .controls {{ display: flex; align-items: center; gap: 14px; margin: 18px 0; }}
  button {{
    background: #2C3E50; color: white; border: none; padding: 10px 22px;
    border-radius: 6px; font-size: 14px; cursor: pointer; transition: background 0.15s;
  }}
  button:hover {{ background: #1a252f; }}
  button:disabled {{ background: #b0b6bb; cursor: not-allowed; }}
  button.reset {{ background: #E67E22; }}
  button.reset:hover {{ background: #c8690f; }}
  .stats {{
    display: flex; gap: 32px; margin-top: 18px; flex-wrap: wrap;
  }}
  .stat {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 14px 20px; min-width: 160px; }}
  .stat .label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.03em; }}
  .stat .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
  .breach {{ color: #E74C3C; }}
  .safe {{ color: #27AE60; }}
  .legend {{ display: flex; gap: 20px; font-size: 13px; margin-top: 10px; color: #555; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 14px; height: 3px; border-radius: 2px; display: inline-block; }}
</style>
</head>
<body>
<div class="container">
  <h1>Monte Carlo Simulated Portfolio Paths ({HORIZON_DAYS}-Day Horizon)</h1>
  <p class="subtitle">
    {NUM_PATHS} simulated trajectories &middot; Student-t distribution (df={df}) &middot;
    99% horizon VaR = {var_pct:.2f}% &middot; Expected Shortfall = {es_pct:.2f}%
  </p>

  <div id="chart-wrap">
    <canvas id="chart"></canvas>
    <div class="legend">
      <span><span class="swatch" style="background:#3498DB"></span> Path stays within VaR</span>
      <span><span class="swatch" style="background:#E74C3C"></span> Path breaches 99% VaR</span>
      <span><span class="swatch" style="background:#2C3E50; height:2px; border-top: 2px dashed #2C3E50; background:transparent"></span> VaR threshold</span>
    </div>
  </div>

  <div class="controls">
    <button id="startBtn">Start Simulation</button>
    <button id="resetBtn" class="reset">Reset</button>
  </div>

  <div class="stats">
    <div class="stat"><div class="label">Paths revealed</div><div class="value" id="statRevealed">0 / {NUM_PATHS}</div></div>
    <div class="stat"><div class="label">Breaching VaR</div><div class="value breach" id="statBreach">0</div></div>
    <div class="stat"><div class="label">Breach rate</div><div class="value" id="statRate">0.0%</div></div>
    <div class="stat"><div class="label">Worst simulated path</div><div class="value breach" id="statWorst">--</div></div>
  </div>
</div>

<script>
const paths = {data_json};
const varThreshold = -{var_pct};
const numDays = {HORIZON_DAYS};
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');

function resize() {{
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}}
window.addEventListener('resize', resize);
resize();

let allValues = paths.flat();
const yMin = Math.min(...allValues, varThreshold) * 1.1;
const yMax = Math.max(...allValues) * 1.1;

function xScale(day, width) {{ return (day / numDays) * (width - 60) + 50; }}
function yScale(val, height) {{
  return height - 40 - ((val - yMin) / (yMax - yMin)) * (height - 70);
}}

function drawAxes(width, height) {{
  ctx.strokeStyle = '#eee';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#999';
  ctx.font = '11px sans-serif';
  for (let g = Math.ceil(yMin/5)*5; g <= yMax; g += 5) {{
    const y = yScale(g, height);
    ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(width - 10, y); ctx.stroke();
    ctx.fillText(g.toFixed(0) + '%', 8, y + 3);
  }}
  // zero line
  ctx.strokeStyle = '#ccc';
  const zeroY = yScale(0, height);
  ctx.beginPath(); ctx.moveTo(50, zeroY); ctx.lineTo(width - 10, zeroY); ctx.stroke();

  // VaR threshold line
  ctx.strokeStyle = '#2C3E50';
  ctx.setLineDash([6, 4]);
  const varY = yScale(varThreshold, height);
  ctx.beginPath(); ctx.moveTo(50, varY); ctx.lineTo(width - 10, varY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#2C3E50';
  ctx.fillText('99% VaR (' + varThreshold.toFixed(2) + '%)', width - 160, varY - 6);
}}

let animId = null;
let progress = 0;

function render(revealFraction) {{
  const rect = canvas.getBoundingClientRect();
  const width = rect.width, height = rect.height;
  ctx.clearRect(0, 0, width, height);
  drawAxes(width, height);

  let revealedCount = Math.floor(paths.length * revealFraction);
  let breachCount = 0;
  let worst = 0;

  for (let i = 0; i < revealedCount; i++) {{
    const path = paths[i];
    const finalVal = path[path.length - 1];
    const breaches = finalVal <= varThreshold;
    if (breaches) {{ breachCount++; worst = Math.min(worst, finalVal); }}

    ctx.strokeStyle = breaches ? 'rgba(231,76,60,0.55)' : 'rgba(52,152,219,0.25)';
    ctx.lineWidth = breaches ? 1.4 : 0.8;
    ctx.beginPath();
    for (let d = 0; d < path.length; d++) {{
      const x = xScale(d, width), y = yScale(path[d], height);
      if (d === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  document.getElementById('statRevealed').textContent = revealedCount + ' / ' + paths.length;
  document.getElementById('statBreach').textContent = breachCount;
  document.getElementById('statRate').textContent = (revealedCount > 0 ? (100*breachCount/revealedCount).toFixed(1) : '0.0') + '%';
  document.getElementById('statWorst').textContent = revealedCount > 0 ? worst.toFixed(2) + '%' : '--';
}}

function animate() {{
  progress += 0.01;
  if (progress >= 1) {{
    progress = 1;
    render(progress);
    document.getElementById('startBtn').disabled = false;
    return;
  }}
  render(progress);
  animId = requestAnimationFrame(animate);
}}

document.getElementById('startBtn').addEventListener('click', () => {{
  document.getElementById('startBtn').disabled = true;
  progress = 0;
  animate();
}});
document.getElementById('resetBtn').addEventListener('click', () => {{
  if (animId) cancelAnimationFrame(animId);
  progress = 0;
  document.getElementById('startBtn').disabled = false;
  render(0);
}});

render(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    config = load_config()
    prices = load_prices()
    rets = simple_returns(prices)
    weights = compute_weights(rets, scheme=config["portfolio"]["weighting_scheme"])
    mc_cfg = config["var"]["monte_carlo"]

    paths = simulate_multiday_paths(
        rets, weights, NUM_PATHS, HORIZON_DAYS,
        distribution=mc_cfg["distribution"], degrees_of_freedom=mc_cfg["degrees_of_freedom"],
        random_seed=mc_cfg["random_seed"],
    )

    final_returns = paths[:, -1]
    alpha = 1 - CONFIDENCE
    var_threshold = -np.quantile(final_returns, alpha)
    tail = final_returns[final_returns <= -var_threshold]
    es = -tail.mean()

    html = build_html(paths, var_threshold, es, mc_cfg["distribution"], mc_cfg["degrees_of_freedom"])
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Saved interactive simulation to {OUTPUT_PATH}")