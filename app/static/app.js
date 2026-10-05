/* Multi-Agent Research Console - ChatGPT-style UI (vanilla JS, talks to /api/v1) */
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

/* ---------------- session & settings ---------------- */
const sessionId = () => {
  let id = localStorage.getItem("sessionId");
  if (!id) {
    id = "s-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem("sessionId", id);
  }
  return id;
};

let visualMode = localStorage.getItem("visualMode") === "1";
let quickMode = localStorage.getItem("quickMode") !== "0";

const visualToggle = $("#visualToggle");
visualToggle.checked = visualMode;
visualToggle.addEventListener("change", () => {
  visualMode = visualToggle.checked;
  localStorage.setItem("visualMode", visualMode ? "1" : "0");
});

const quickToggle = $("#quickToggle");
quickToggle.checked = quickMode;
quickToggle.addEventListener("change", () => {
  quickMode = quickToggle.checked;
  localStorage.setItem("quickMode", quickMode ? "1" : "0");
});

const newChatBtn = $("#newChatBtn");
newChatBtn.addEventListener("click", () => {
  localStorage.removeItem("sessionId");
  $("#chatMessages").innerHTML = "";
  chatHint("Ask anything. Follow-ups use quick answers with memory of this chat.");
  welcome();
});

const deepToggle = $("#deepToggle");

/* ---------------- tabs ---------------- */
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("tab--active"));
    $$(".panel").forEach((p) => p.classList.remove("panel--active"));
    tab.classList.add("tab--active");
    $("#panel-" + tab.dataset.tab).classList.add("panel--active");
    if (tab.dataset.tab === "history") loadHistory();
    if (tab.dataset.tab === "settings") loadSettings();
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
      ? `LIVE MODE\u00b7${(h.provider || "openai").toUpperCase()}`
      : "FALLBACK MODE (no API keys)";
    mode.classList.toggle("badge--amber", !h.llm_configured);
    mode.classList.toggle("badge--green", !!h.llm_configured);
  } catch {
    $("#healthBadge").classList.add("badge--red");
    $("#healthBadge").textContent = "API DOWN";
  }
}

/* ---------------- chat ---------------- */
const chat = $("#chatMessages");
let msgSeq = 0;
let suggestionBar = null;

function bubble(kind, html) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.innerHTML = html;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function removeSuggestions() {
  if (suggestionBar) { suggestionBar.remove(); suggestionBar = null; }
}

function addSuggestions(question) {
  removeSuggestions();
  const ideas = [
    `Break ${question.slice(0, 60)} into decisions`,
    "Summarize the main risks",
    "What data would change your answer?",
  ];
  const el = document.createElement("div");
  el.className = "chips";
  el.innerHTML = ideas.map((t) => `<button class="chip-btn">${esc(t)}</button>`).join("");
  el.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => ask(btn.textContent));
  });
  chat.appendChild(el);
  suggestionBar = el;
  chat.scrollTop = chat.scrollHeight;
}

function workingBubble() {
  const el = bubble("msg-ai working", `<div class="msg-bubble msg-bubble--ai"><div class="stage"><span class="spinner"></span><span class="stage-text">Starting the research pipeline\u2026</span></div></div>`);
  return { el, text: el.querySelector(".stage-text") };
}

function chatHint(text) {
  $("#chatHint").innerHTML = text;
}

async function ask(question, forceFull = deepToggle.checked) {
  const q = question.trim();
  if (!q) { $("#chatInput").focus(); return; }

  bubble("msg-user", `<div class="msg-bubble msg-bubble--user">${esc(q)}</div>`);
  $("#chatInput").value = "";
  $("#chatInput").disabled = true;
  $("#sendBtn").disabled = true;
  const w = workingBubble();
  removeSuggestions();

  let chunkShown = false;
  let chunkText = "";
  let chunkEl = null;
  let finalData = null;

  const showChunk = (text) => {
    if (!chunkEl) {
      chunkEl = bubble("msg-ai", `<div class="msg-bubble msg-bubble--ai"><div class="markdown"></div><span class="cursor"></span></div>`);
    }
    chunkShown = true;
    chunkText += text;
    const render = () => {
      const md = chunkEl.querySelector(".markdown");
      md.innerHTML = window.marked ? window.marked.parse(chunkText) : esc(chunkText);
      chat.scrollTop = chat.scrollHeight;
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(render);
    else render();
  };

  const finalizeChunks = () => {
    if (chunkEl) { chunkEl.querySelector(".cursor")?.remove(); }
    if (chunkShown) { addSuggestions(q); reenable(); return true; }
    return false;
  };

  const reenable = () => {
    $("#chatInput").disabled = false;
    $("#sendBtn").disabled = false;
    $("#chatInput").focus();
    chatHint("Sent. Ask a follow-up and press <b>Enter</b> \u2014 it remembers this conversation.");
  };

  try {
    const useQuick = quickMode && !deepToggle.checked;
    const res = await fetch(`${API}/agent/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_request: q, session_id: sessionId(), quick: useQuick }),
    });

    if (!res.ok || !res.body) {
      w.el.remove();
      bubble("msg-ai", `<div class="msg-bubble msg-bubble--error">Request failed (${res.status}): ${esc(await res.text())}</div>`);
      reenable();
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const handleEvent = (ev) => {
      if (ev.type === "status") {
        w.text.textContent = ev.text || "Working\u2026";
      } else if (ev.type === "chunk") {
        w.el.remove();
        showChunk(ev.text || "");
      } else if (ev.type === "answer") {
        w.el.remove();
        finalData = ev.data || {};
        if (finalizeChunks()) return;
        createAnswer(finalData);
        addSuggestions(q);
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawBlock = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        for (const line of rawBlock.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try { handleEvent(JSON.parse(line.slice(6))); } catch { /* ignore */ }
        }
      }
    }
    if (!w.el.isConnected && !chunkShown && !finalData) {
      reenable();
    } else if (!finalData && !chunkShown) {
      w.el.remove();
      bubble("msg-ai", `<div class="msg-bubble msg-bubble--error">Stream ended with no answer.</div>`);
      reenable();
    } else {
      reenable();
    }
  } catch (e) {
    w.el.remove();
    bubble("msg-ai", `<div class="msg-bubble msg-bubble--error">Could not reach server: ${esc(String(e))}</div>`);
    reenable();
  }
}

$("#chatForm").addEventListener("submit", (e) => { e.preventDefault(); ask($("#chatInput").value); });

$("#chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask($("#chatInput").value);
  }
});

function welcome() {
  bubble("msg-ai", `<div class="msg-bubble msg-bubble--hint">Hi! Ask any analytical question and press <b>Enter</b>. Follow-ups get <b>quick answers</b> that remember this chat \u2014 toggle <b>Deep research</b> for a full multi-agent report.</div>`);
}
welcome();
$("#chatInput").focus();

/* ---------------- answer rendering ---------------- */
function statusChip(s) {
  return `<span class="status-chip ${esc(s)}">${esc(s)}</span>`;
}

let ansSeq = 0;

function tldr(report) {
  if (!report) return "";
  const plain = report
    .replace(/[#*_>`[\]-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const first = plain.split(/\.(?: |$)/).slice(0, 2).join(". ") + ".";
  const words = first.split(/\s+/).slice(0, 40).join(" ");
  return words;
}

function createAnswer(data) {
  const el = document.createElement("div");
  el.className = "msg msg-ai";
  el._id = "ans" + ++ansSeq;
  el.innerHTML = `
    <div class="msg-bubble msg-bubble--ai">
      ${data.quick ? `<div class="answer-head"><div class="answer-title">Quick answer</div></div>` : ""}
      ${!data.quick ? `<div class="answer-head">
        <div class="answer-title">Answer</div>
        <div class="answer-tabs">
          <button class="ans-tab ${visualMode ? "" : "ans-tab--active"}" data-view="text">Text</button>
          <button class="ans-tab ${visualMode ? "ans-tab--active" : ""}" data-view="visual">Visual</button>
        </div>
      </div>` : ""}
      ${data.report && !data.quick ? `<div class="tldr">TL;DR\u2014${esc(tldr(data.report))}</div>` : ""}
      <div class="answer-view ${visualMode ? "" : "answer-view--active"} ${data.quick ? "answer-view--active" : ""}" data-view="text">${buildTextHTML(data)}</div>
      ${data.quick ? "" : `<div class="answer-view answer-view--visual ${visualMode ? "answer-view--active" : ""}" data-view="visual"></div>`}
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
  if (visualMode && !data.quick) renderVisual(el, data);
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
    ${d.quick ? `<div class="markdown">${reportHtml}</div>` : `
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

    ${d.report ? `<div class="card"><h3>Report</h3><div class="markdown">${reportHtml}</div></div>` : ""}`}`;
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
    <div class="card tldr">&#128172; Plain-language recap\u2014${esc(tldr(d.report))}</div>
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
  charts.push(gauge);

  const okCount = ev.filter((e) => e.sufficient).length;
  const thinCount = ev.length - okCount;
  const donut = new Chart(document.getElementById(`${el._id}-donut`), {
    type: "doughnut",
    data: { labels: ["Sufficient", "Thin"], datasets: [{ data: [okCount, thinCount], backgroundColor: [CHART_COLORS.ok, CHART_COLORS.thin], borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false },
  });
  charts.push(donut);

  const labels = ev.length ? ev.map((e) => e.subtask || "?") : ["No evidence"];
  const scores = ev.length ? ev.map((e) => Math.round((e.score ?? 0) * 100)) : [0];
  const colors = ev.length ? ev.map((e) => (e.sufficient ? CHART_COLORS.ok : CHART_COLORS.thin)) : ["#4f8cff"];
  const evidenceBar = new Chart(document.getElementById(`${el._id}-evidence`), {
    type: "bar",
    data: { labels, datasets: [{ data: scores, backgroundColor: colors, borderRadius: 6 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { max: 100, ticks: { color: "#93a1bd" } }, y: { ticks: { color: "#e8edf7" } } }, plugins: { legend: { display: false } } },
  });
  charts.push(evidenceBar);

  const sourcesBar = new Chart(document.getElementById(`${el._id}-sources`), {
    type: "bar",
    data: { labels, datasets: [{ data: ev.length ? ev.map((e) => e.sources ?? 0) : [0], backgroundColor: CHART_COLORS.accent, borderRadius: 6 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: "#93a1bd" }, beginAtZero: true }, y: { ticks: { color: "#e8edf7" } } }, plugins: { legend: { display: false } } },
  });
  charts.push(sourcesBar);

  el._visualCharts = charts;
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

/* ---------------- settings (in-app key vault) ---------------- */
async function loadSettings() {
  const summary = $("#settingsSummary");
  try {
    const s = await (await fetch(`${API}/settings`)).json();
    $("#providerSelect").value = s.provider === "gemini" ? "gemini" : "openai";
    $("#openaiKeyField").placeholder = s.openai_api_key ? `current: ${s.openai_api_key}` : "sk-...";
    $("#geminiKeyField").placeholder = s.gemini_api_key ? `current: ${s.gemini_api_key}` : "AQ.... / AIza...";
    $("#tavilyKeyField").placeholder = s.tavily_api_key ? `current: ${s.tavily_api_key}` : "tvly-...";
    summary.hidden = false;
    summary.innerHTML = `
      <span class="chip ${s.llm_configured ? "ok" : "thin"}">${s.llm_configured ? "LIVE\u00b7" + (s.provider || "openai").toUpperCase() : "FALLBACK MODE"}</span>
      <span>${s.web_search_configured ? "web search live" : "web search off"}</span>`;
  } catch (e) {
    summary.hidden = false;
    summary.innerHTML = `<span class="chip thin">could not load settings: ${esc(String(e))}</span>`;
  }
}

async function saveSettings() {
  const btn = $("#saveSettingsBtn");
  btn.disabled = true;
  $("#settingsStatus").textContent = "saving\u2026";
  try {
    const resp = await fetch(`${API}/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        llm_provider: $("#providerSelect").value,
        openai_api_key: $("#openaiKeyField").value,
        gemini_api_key: $("#geminiKeyField").value,
        tavily_api_key: $("#tavilyKeyField").value,
      }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    $("#openaiKeyField").value = "";
    $("#geminiKeyField").value = "";
    $("#tavilyKeyField").value = "";
    $("#settingsStatus").textContent = "saved";
    loadHealth();
    await loadSettings();
  } catch (e) {
    $("#settingsStatus").textContent = "error: " + String(e);
  } finally {
    btn.disabled = false;
  }
}
$("#saveSettingsBtn").addEventListener("click", saveSettings);

/* ---------------- sandbox ---------------- */
async function runSandbox() {
  const code = $("#code").value.trim();
  if (!code) return;
  $("#analyzeBtn").disabled = true;
  $("#analyzeStatus").textContent = "running\u2026";
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