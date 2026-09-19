/* Multi-Agent Research Console - UI logic (vanilla JS, talks to /api/v1) */
const API = "/api/v1";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const esc = (s) =>
  String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const fmtTime = (iso) => new Date(iso).toLocaleString();

/* ---------------- tabs ---------------- */
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("tab--active"));
    $$(".panel").forEach((p) => p.classList.remove("panel--active"));
    tab.classList.add("tab--active");
    $("#panel-" + tab.dataset.tab).classList.add("panel--active");
    if (tab.dataset.tab === "history") loadHistory();
  });
});

/* ---------------- health ---------------- */
async function loadHealth() {
  try {
    const h = await (await fetch(`${API}/health`)).json();
    const badge = $("#healthBadge");
    badge.classList.add("badge--green");
    badge.textContent = h.status === "ok" ? "API OK" : "API DEGRADED";
    const mode = $("#modeBadge");
    mode.textContent = h.llm_configured
      ? "LIVE MODE"
      : "FALLBACK MODE (no API keys)";
    mode.classList.toggle("badge--amber", !h.llm_configured);
    mode.classList.toggle("badge--green", !!h.llm_configured);
  } catch {
    const badge = $("#healthBadge");
    badge.classList.add("badge--red");
    badge.textContent = "API DOWN";
  }
}

/* ---------------- chat loop ---------------- */
const chat = $("#chatMessages");

function bubble(kind, html) {
  const el = document.createElement("div");
  el.className = "msg msg-" + kind;
  el.innerHTML = html;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function bubbleWork() {
  const el = bubble("ai working", '<span class="spinner"></span><span>Running pipeline (plan &#8594; research &#8594; coverage &#8594; analyze &#8594; report &#8594; critique)&#8230;</span>');
  return el;
}

async function ask(question) {
  const q = question.trim();
  if (!q) { $("#chatInput").focus(); return; }

  bubble("user", `<div class="msg-bubble">${esc(q)}</div>`);
  $("#chatInput").value = "";
  $("#chatInput").disabled = true;
  $("#sendBtn").disabled = true;
  const working = bubbleWork();

  try {
    const res = await fetch(`${API}/agent/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_request: q }),
    });
    const data = res.ok ? await res.json() : { status: "failed", error: await res.text() };
    working.remove();
    if (data.status === "failed") {
      bubble("ai", `<div class="msg-bubble msg-bubble--error">Request failed: ${esc(data.error || "unknown error")}</div>`);
    } else {
      createAnswer(data);
    }
  } catch (e) {
    working.remove();
    bubble("ai", `<div class="msg-bubble msg-bubble--error">Could not reach server: ${esc(String(e))}</div>`);
  } finally {
    $("#chatInput").disabled = false;
    $("#sendBtn").disabled = false;
    $("#chatInput").focus();
    $("#chatHint").innerHTML = "Done. Ask another query and press <b>Enter</b>, or <b>Shift+Enter</b> for a new line.";
  }
}

$("#chatForm").addEventListener("submit", (e) => {
  e.preventDefault();
  ask($("#chatInput").value);
});

$("#chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask($("#chatInput").value);
  }
});

bubble("ai", `<div class="msg-bubble msg-bubble--hint">Hi! Ask any analytical question below and press <b>Enter</b> to run the multi-agent pipeline.</div>`);
$("#chatInput").focus();

/* ---------------- visual mode ---------------- */
const visualToggle = $("#visualToggle");
let visualMode = localStorage.getItem("visualMode") === "1";
visualToggle.checked = visualMode;
visualToggle.addEventListener("change", () => {
  visualMode = visualToggle.checked;
  localStorage.setItem("visualMode", visualMode ? "1" : "0");
});

/* ---------------- answer rendering ---------------- */
function statusChip(s) {
  return `<span class="status-chip ${esc(s)}">${esc(s)}</span>`;
}

let ansSeq = 0;

function createAnswer(data) {
  const el = document.createElement("div");
  el.className = "msg msg-ai";
  el._id = "ans" + ++ansSeq;
  el.innerHTML = `
    <div class="msg-bubble msg-bubble--ai">
      <div class="answer-head">
        <div class="answer-title">Answer</div>
        <div class="answer-tabs">
          <button class="ans-tab ${visualMode ? "" : "ans-tab--active"}" data-view="text">Text</button>
          <button class="ans-tab ${visualMode ? "ans-tab--active" : ""}" data-view="visual">Visual</button>
        </div>
      </div>
      <div class="answer-view ${visualMode ? "" : "answer-view--active"}" data-view="text">${buildTextHTML(data)}</div>
      <div class="answer-view answer-view--visual ${visualMode ? "answer-view--active" : ""}" data-view="visual"></div>
    </div>`;

  el.querySelectorAll(".ans-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      el.querySelectorAll(".ans-tab").forEach((b) => b.classList.toggle("ans-tab--active", b === btn));
      el.querySelectorAll(".answer-view").forEach((v) =>
        v.classList.toggle("answer-view--active", v.dataset.view === btn.dataset.view));
      if (btn.dataset.view === "visual") renderVisual(el, data);
    });
  });

  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  if (visualMode) renderVisual(el, data);
}

function buildTextHTML(d) {
  const coverage = Math.round((d.coverage_score ?? 0) * 100);
  const covClass = coverage >= 60 ? "ok" : "thin";

  const meta = [
    ["Status", statusChip(d.status)],
    ["Approved", d.approved ? "&#10003; yes" : "no"],
    ["Iterations", d.iterations],
    ["Research depth", d.research_depth],
    ["Coverage", coverage + "%"],
    ["Sources", (d.research_sources || []).length],
  ].map(([k, v]) => `<div class="meta"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`).join("");

  const evidence = (d.evidence || []).map((e) => {
    const score = Math.round((e.score ?? 0) * 100);
    return `<div class="evidence-row">
      <div>${esc(e.subtask)}</div>
      <div class="ev-score">${score}% &middot; ${e.sources ?? 0} src</div>
      <div class="chip ${e.sufficient ? "ok" : "thin"}">${e.sufficient ? "OK" : "THIN"}</div>
    </div>`;
  }).join("") || '<div class="muted">No per-sub-task evidence recorded.</div>';

  const feedback = (d.critique_feedback || []).map((f) => `<li>${esc(f)}</li>`).join("")
    || '<li class="muted">No feedback.</li>';

  const charts = (d.charts || []).map((b64) =>
    `<img src="data:image/png;base64,${b64}" alt="chart" />`).join("");

  const reportHtml = (window.marked && d.report)
    ? window.marked.parse(d.report)
    : `<pre class="code-block">${esc(d.report || "(no report)")}</pre>`;

  return `
    <div class="grid cards-meta">${meta}</div>

    <div class="grid two">
      <div class="card">
        <h3>Evidence &amp; Coverage <span class="chip ${covClass}">${coverage}%</span></h3>
        <div class="bar"><div class="bar-fill" style="width:${coverage}%"></div></div>
        ${evidence}
      </div>
      <div class="card">
        <h3>Critique feedback</h3>
        <ul class="list">${feedback}</ul>
      </div>
    </div>

    ${charts ? `<div class="card"><h3>Charts</h3><div class="charts">${charts}</div></div>` : ""}

    <details class="card"><summary>Analysis code</summary>
      <pre class="code-block">${esc(d.analysis_code || "(none)")}</pre>
    </details>
    <details class="card"><summary>Analysis output</summary>
      <pre class="code-block">${esc(d.analysis_output + (d.analysis_result ? "\n\nresult:\n" + JSON.stringify(d.analysis_result, null, 2) : ""))}</pre>
    </details>

    ${d.report ? `<div class="card"><h3>Report</h3><div class="markdown">${reportHtml}</div></div>` : ""}`;
}

/* ---------------- visual rendering (Chart.js) ---------------- */
const CHART_COLORS = { ok: "rgba(56,193,114,.85)", thin: "rgba(229,83,75,.85)", accent: "#4f8cff" };

function summaryChips(d) {
  const ev = d.evidence || [];
  const okCount = ev.filter((e) => e.sufficient).length;
  const thinCount = ev.length - okCount;
  const coverage = Math.round((d.coverage_score ?? 0) * 100);
  return `
    <div class="grid cards-meta">
      <div class="meta"><div class="k">Coverage</div><div class="v">${coverage}%</div></div>
      <div class="meta"><div class="k">Sub-tasks OK</div><div class="v">${okCount}</div></div>
      <div class="meta"><div class="k">Sub-tasks THIN</div><div class="v">${thinCount}</div></div>
      <div class="meta"><div class="k">Research depth</div><div class="v">${d.research_depth}</div></div>
      <div class="meta"><div class="k">Iterations</div><div class="v">${d.iterations}</div></div>
      <div class="meta"><div class="k">Status</div><div class="v">${statusChip(d.status)}</div></div>
    </div>`;
}

function renderVisual(el, d) {
  const view = el.querySelector('[data-view="visual"]');
  if (!view) return;
  if (!window.Chart) {
    view.innerHTML = `<div class="card">Chart library unavailable (offline). Showing numbers:<pre class="code-block">${esc(JSON.stringify({ coverage_score: d.coverage_score, evidence: d.evidence }, null, 2))}</pre></div>`;
    return;
  }

  if (el._visualDone) { view.innerHTML = el._visualSaved; el._visualCharts.forEach((c) => c.destroy()); el._visualCharts = []; }
  const ev = d.evidence || [];
  const covPct = Math.round((d.coverage_score ?? 0) * 100);

  view.innerHTML = `
    ${summaryChips(d)}
    <div class="grid two">
      <div class="card"><h3>Coverage score</h3><div class="chart-box"><canvas id="${el._id}-gauge"></canvas></div></div>
      <div class="card"><h3>Sub-tasks: sufficient vs thin</h3><div class="chart-box"><canvas id="${el._id}-donut"></canvas></div></div>
    </div>
    <div class="card"><h3>Evidence score per sub-task (0-100)</h3><div class="chart-box chart-box--tall"><canvas id="${el._id}-evidence"></canvas></div></div>
    <div class="card"><h3>Sources per sub-task</h3><div class="chart-box chart-box--tall"><canvas id="${el._id}-sources"></canvas></div></div>
    ${(d.charts || []).length ? `<div class="card"><h3>Analysis charts</h3><div class="charts">${(d.charts || []).map((b64) => `<img src="data:image/png;base64,${b64}" alt="chart" />`).join("")}</div></div>` : ""}`;

  el._visualSaved = view.innerHTML;
  el._visualDone = true;

  const charts = [];
  const gauge = new Chart(document.getElementById(`${el._id}-gauge`), {
    type: "doughnut",
    data: { datasets: [{ data: [covPct, 100 - covPct], backgroundColor: [CHART_COLORS.ok, "rgba(255,255,255,.06)"], borderWidth: 0 }] },
    options: { cutout: "72%", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, centerText: { text: covPct + "%" } } },
    plugins: [centerTextPlugin],
  });

  const okCount = ev.filter((e) => e.sufficient).length;
  const thinCount = ev.length - okCount;
  const donut = new Chart(document.getElementById(`${el._id}-donut`), {
    type: "doughnut",
    data: { labels: ["Sufficient", "Thin"], datasets: [{ data: [okCount, thinCount], backgroundColor: [CHART_COLORS.ok, CHART_COLORS.thin], borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false },
  });

  const labels = ev.length ? ev.map((e) => e.subtask || "?") : ["No evidence"];
  const scores = ev.length ? ev.map((e) => Math.round((e.score ?? 0) * 100)) : [0];
  const colors = ev.length ? ev.map((e) => (e.sufficient ? CHART_COLORS.ok : CHART_COLORS.thin)) : ["#4f8cff"];
  const evidence = new Chart(document.getElementById(`${el._id}-evidence`), {
    type: "bar",
    data: { labels, datasets: [{ data: scores, backgroundColor: colors, borderRadius: 6 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { max: 100, ticks: { color: "#93a1bd" } }, y: { ticks: { color: "#e8edf7" } } }, plugins: { legend: { display: false } } },
  });

  const sources = new Chart(document.getElementById(`${el._id}-sources`), {
    type: "bar",
    data: { labels, datasets: [{ data: ev.length ? ev.map((e) => e.sources ?? 0) : [0], backgroundColor: CHART_COLORS.accent, borderRadius: 6 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: "#93a1bd" }, beginAtZero: true } , y: { ticks: { color: "#e8edf7" } } }, plugins: { legend: { display: false } } },
  });

  el._visualCharts = [gauge, donut, evidence, sources];
  chat.scrollTop = chat.scrollHeight;
}

const centerTextPlugin = {
  id: "centerText",
  afterDraw(chart) {
    const text = chart.options.plugins.centerText.text;
    if (!text) return;
    const { ctx } = chart;
    const meta = chart.getDatasetMeta(0);
    const x = meta.data[0] ? meta.data[0].x : chart.width / 2;
    const y = meta.data[0] ? meta.data[0].y : chart.height / 2;
    ctx.save();
    ctx.font = "700 30px 'Segoe UI', sans-serif";
    ctx.fillStyle = "#e8edf7";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
    ctx.restore();
  },
};

/* ---------------- task history ---------------- */
async function loadHistory() {
  const rows = $("#historyBody");
  rows.innerHTML = '<tr><td colspan="4" class="muted">Loading&#8230;</td></tr>';
  try {
    const data = await (await fetch(`${API}/tasks`)).json();
    const items = data.value || data;
    if (!items.length) {
      rows.innerHTML = '<tr><td colspan="4" class="muted">No tasks yet.</td></tr>';
      return;
    }
    rows.innerHTML = items.map((t) => `<tr data-id="${esc(t.task_id)}">
        <td>${statusChip(t.status)}</td>
        <td>${esc(t.task_id.slice(0, 8))}&#8230;</td>
        <td>${esc(t.user_request)}</td>
        <td class="muted">${fmtTime(t.created_at)}</td>
      </tr>`).join("");
    rows.querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", async () => {
        const d = await (await fetch(`${API}/tasks/${tr.dataset.id}`)).json();
        createAnswer(d);
        $$(".tab").forEach((t) => t.classList.remove("tab--active"));
        $$(".panel").forEach((p) => p.classList.remove("panel--active"));
        document.querySelector('.tab[data-tab="chat"]').classList.add("tab--active");
        $("#panel-chat").classList.add("panel--active");
      });
    });
  } catch (e) {
    rows.innerHTML = `<tr><td colspan="4" class="muted">${esc(String(e))}</td></tr>`;
  }
}
$("#refreshHistoryBtn").addEventListener("click", loadHistory);

/* ---------------- sandbox ---------------- */
async function runSandbox() {
  const code = $("#code").value.trim();
  if (!code) return;
  $("#analyzeBtn").disabled = true;
  $("#analyzeStatus").textContent = "running&#8230;";
  try {
    const d = await (await fetch(`${API}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    })).json();
    let out = d.stdout || "";
    if (d.stderr) out += "\n--- stderr ---\n" + d.stderr;
    if (d.result !== undefined && d.result !== null) out += "\n--- result ---\n" + JSON.stringify(d.result, null, 2);
    if (d.error) out += "\n--- error ---\n" + d.error;
    $("#analyzeOut").textContent = out || "(no output)";
    $("#analyzeCharts").innerHTML = (d.charts || [])
      .map((b64) => `<img src="data:image/png;base64,${b64}" alt="chart" />`).join("");
    $("#analyzeStatus").textContent = d.ok ? "ok" : "failed";
  } catch (e) {
    $("#analyzeOut").textContent = String(e);
    $("#analyzeStatus").textContent = "error";
  } finally {
    $("#analyzeBtn").disabled = false;
  }
}
$("#analyzeBtn").addEventListener("click", runSandbox);

loadHealth();
setInterval(loadHealth, 15000);