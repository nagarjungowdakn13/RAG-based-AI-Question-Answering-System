/* ─────────────────────────────────────────────────────────────
   RAG QA Dashboard — frontend logic.
   Vanilla JS + Chart.js. Talks to FastAPI on the same origin.
   ───────────────────────────────────────────────────────────── */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const fmt2 = (x) => (x == null ? "–" : Number(x).toFixed(2));
const basename = (p) => (p ? p.split(/[\\/]/).pop() : "unknown");
const escapeHTML = (s) =>
  String(s).replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[m]));

// ─── theme toggle ──────────────────────────────────────────────
const root = document.documentElement;
const savedTheme = localStorage.getItem("rag-theme") || "light";
root.dataset.theme = savedTheme;
$("#theme-icon").textContent = savedTheme === "dark" ? "☾" : "☀";

$("#theme-toggle").addEventListener("click", () => {
  const next = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  localStorage.setItem("rag-theme", next);
  $("#theme-icon").textContent = next === "dark" ? "☾" : "☀";
  applyChartTheme();
});

// ─── Chart.js global look ───────────────────────────────────────
Chart.defaults.font.family = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
Chart.defaults.font.size = 11.5;
Chart.defaults.plugins.legend.display = false;

// Earth-tone chart palette: terracotta, sage, olive, ochre, brick.
// Keep the names so the rest of the file still reads coherently.
const PRIMARY = "#b95a35"; // terracotta
const ACCENT  = "#7c9473"; // sage
const SUCCESS = "#6f8b4a"; // olive
const WARN    = "#c69b3f"; // ochre
const DANGER  = "#a14b3a"; // brick

function themeColors() {
  const dark = root.dataset.theme === "dark";
  return {
    text:  dark ? "#ece4d3" : "#2b2620",                       // parchment / ink
    muted: dark ? "#9d8e76" : "#7a6f5e",                       // warm gray
    grid:  dark ? "rgba(232,213,182,.10)" : "rgba(122,111,94,.18)",
  };
}
function applyChartTheme() {
  const c = themeColors();
  Chart.defaults.color = c.muted;
  [confChart, timelineChart, evalChart].forEach((ch) => {
    if (!ch) return;
    if (ch.options.scales) {
      Object.values(ch.options.scales).forEach((s) => {
        s.grid = { ...(s.grid || {}), color: c.grid };
        s.ticks = { ...(s.ticks || {}), color: c.muted };
      });
    }
    ch.update();
  });
}

// ─── fetch with timeout ────────────────────────────────────────
async function fetchT(url, options = {}, timeoutMs = 90000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ─── status pill + KPIs ─────────────────────────────────────────
async function refreshStats() {
  try {
    const r = await fetchT("/stats", {}, 10000);
    if (!r.ok) throw new Error("stats failed");
    const j = await r.json();
    paintKpis(j);
    paintCharts(j);
    paintTopSources(j.top_sources);
    paintRecent(j.recent_queries);
    paintStatusPill(j);
    paintSourceFilter(j.indexed_sources || []);
    updateChatEmptyState(j);
  } catch {
    const pill = $("#status-pill");
    pill.className = "pill pill-err";
    $("#status-text").textContent = "API offline";
  }
}

function paintStatusPill(j) {
  const pill = $("#status-pill");
  pill.className = "pill pill-ok";
  const parts = [
    `${j.index_size} chunks`,
    j.embedding_backend,
    j.llm_backend,
  ];
  if (j.retrieval_mode) parts.push(`${j.retrieval_mode}${j.mmr_enabled ? "+mmr" : ""}`);
  if (j.cache && j.cache.maxsize > 0) {
    parts.push(`cache ${j.cache.hits}/${j.cache.hits + j.cache.misses}`);
  }
  $("#status-text").textContent = parts.join(" · ");
}

// ─── source filter chips ───────────────────────────────────────
const activeSourceFilter = new Set();

function paintSourceFilter(sources) {
  const row = $("#source-filter-row");
  const container = $("#src-filter-chips");
  if (!sources || sources.length === 0) {
    row.classList.add("hidden");
    container.innerHTML = "";
    activeSourceFilter.clear();
    return;
  }
  row.classList.remove("hidden");
  // Drop any active filter that no longer exists in the index.
  for (const name of [...activeSourceFilter]) {
    if (!sources.includes(name)) activeSourceFilter.delete(name);
  }
  container.innerHTML = sources
    .map((name) => {
      const active = activeSourceFilter.has(name) ? " active" : "";
      return `<button type="button" class="src-chip${active}" data-src="${escapeHTML(name)}">${escapeHTML(name)}</button>`;
    })
    .join("");
  container.querySelectorAll(".src-chip").forEach((chip) => {
    chip.onclick = () => {
      const name = chip.dataset.src;
      if (activeSourceFilter.has(name)) activeSourceFilter.delete(name);
      else activeSourceFilter.add(name);
      chip.classList.toggle("active");
    };
  });
}

$("#src-filter-clear").addEventListener("click", () => {
  activeSourceFilter.clear();
  $$(".src-chip").forEach((c) => c.classList.remove("active"));
});

function paintKpis(j) {
  $("#kpi-chunks").textContent = j.index_size;
  $("#kpi-docs").textContent = `${j.documents} document${j.documents === 1 ? "" : "s"}`;
  $("#kpi-queries").textContent = j.queries.total;
  $("#kpi-answered").textContent = j.queries.answered;
  $("#kpi-abstained").textContent = j.queries.abstained;
  $("#kpi-confidence").textContent = fmt2(j.queries.avg_confidence);
  $("#kpi-backend").textContent = j.embedding_backend;
  $("#kpi-llm").textContent = `LLM: ${j.llm_backend}`;
}

// ─── charts ─────────────────────────────────────────────────────
let confChart, timelineChart, evalChart;

function paintCharts(j) {
  const c = themeColors();
  const baseScales = {
    x: { grid: { display: false }, ticks: { color: c.muted } },
    y: { beginAtZero: true, ticks: { precision: 0, color: c.muted }, grid: { color: c.grid } },
  };

  // Confidence histogram
  const confLabels = j.confidence_distribution.map((b) => b.bin);
  const confData = j.confidence_distribution.map((b) => b.count);
  // Confidence buckets in earth tones: low → high.
  // brick → ochre → mustard → olive → sage.
  const confColors = ["#a14b3a", "#c69b3f", "#d4a85a", "#6f8b4a", "#7c9473"];

  if (!confChart) {
    confChart = new Chart($("#chart-confidence"), {
      type: "bar",
      data: { labels: confLabels, datasets: [{ data: confData, backgroundColor: confColors, borderRadius: 8 }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },          // no re-animation on refresh
        animations: { colors: false, x: false, y: false },
        plugins: { tooltip: { mode: "index" } },
        scales: baseScales,
      },
    });
  } else {
    confChart.data.labels = confLabels;
    confChart.data.datasets[0].data = confData;
    confChart.update("none");                 // silent update, no flicker
  }

  // Timeline
  const tlLabels = j.timeline.map((t) => {
    const d = new Date(t.hour);
    return `${String(d.getHours()).padStart(2, "0")}:00`;
  });
  const tlData = j.timeline.map((t) => t.count);
  if (!timelineChart) {
    const ctx = $("#chart-timeline").getContext("2d");
    // Soft terracotta wash under the line, fading to nothing — like a
     // watercolor stain. No neon halo.
    const grad = ctx.createLinearGradient(0, 0, 0, 220);
    grad.addColorStop(0, "rgba(185, 90, 53, .28)");
    grad.addColorStop(1, "rgba(185, 90, 53, 0)");
    timelineChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: tlLabels,
        datasets: [{
          data: tlData,
          borderColor: PRIMARY,
          backgroundColor: grad,
          fill: true,
          tension: 0.4,
          pointRadius: 2,
          pointBackgroundColor: PRIMARY,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        animations: { colors: false, x: false, y: false },
        plugins: { tooltip: { mode: "index" } },
        scales: baseScales,
      },
    });
  } else {
    timelineChart.data.labels = tlLabels;
    timelineChart.data.datasets[0].data = tlData;
    timelineChart.update("none");
  }
}

function paintTopSources(items) {
  const ul = $("#top-sources");
  if (!items || items.length === 0) {
    ul.innerHTML = '<li class="empty-row">No queries yet</li>';
    return;
  }
  const max = Math.max(...items.map((i) => i.count));
  ul.innerHTML = items
    .map((i) => `
      <li>
        <span class="src-name">${escapeHTML(i.source)}</span>
        <span class="src-bar"><span style="width:${(i.count / max) * 100}%"></span></span>
        <span class="src-count">${i.count}</span>
      </li>`)
    .join("");
}

function paintRecent(items) {
  const ul = $("#recent-queries");
  if (!items || items.length === 0) {
    ul.innerHTML = '<li class="empty-row">No queries yet</li>';
    return;
  }
  ul.innerHTML = items
    .map((q) => {
      const conf = q.confidence_score || 0;
      const cls = q.confident ? (conf >= 0.55 ? "conf-high" : "conf-mid") : "conf-low";
      const label = q.confident ? (conf >= 0.55 ? "high" : "moderate") : "abstained";
      const t = q.ts ? new Date(q.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
      return `
        <li>
          <div class="rq-q">${escapeHTML(q.question || "")}</div>
          <div class="rq-meta">
            <span class="conf-badge ${cls}">● ${label} · ${conf.toFixed(2)}</span>
            <span>${t}</span>
          </div>
        </li>`;
    })
    .join("");
}

// ─── chat empty-state: show Init button when index is empty ────
const SUGGESTIONS = [
  "Who coined the term machine learning?",
  "How does RAG reduce hallucination?",
  "Why do Transformers need positional encodings?",
  "What is the capital of France?",
];

function updateChatEmptyState(j) {
  const empty = chat.querySelector(".empty");
  if (!empty) return; // user has already started chatting

  if (j.index_size === 0) {
    if (empty.querySelector("#quick-init")) return; // already rendered
    empty.innerHTML = `
      <p style="font-size:15px;margin-bottom:6px"><strong>📚 The FAISS index is empty</strong></p>
      <p class="hint" style="margin-bottom:16px">Click below to ingest the bundled example dataset (3 docs, ~21 chunks).</p>
      <button class="primary" id="quick-init" style="font-size:14px;padding:11px 22px">🚀 Initialize with example docs</button>
      <p class="hint" style="margin-top:14px">Or use the Ingest panel on the left for your own files.</p>
    `;
    const btn = document.getElementById("quick-init");
    btn.onclick = async () => {
      btn.disabled = true;
      btn.innerHTML = '<span class="spin"></span>Ingesting…';
      try {
        const res = await ingestPath("data/docs");
        btn.innerHTML = `✓ ${res.total_chunks} chunks added`;
        await refreshStats();
      } catch (e) {
        btn.innerHTML = `Failed: ${e.message}`;
        btn.disabled = false;
      }
    };
  } else {
    // Non-empty index: ensure suggestions are present AND have handlers.
    if (empty.dataset.populated !== "yes") {
      empty.innerHTML = `
        <p>Ingested docs are ready. Click a starter or type your own.</p>
        <div class="suggestions">
          ${SUGGESTIONS.map((s) => `<button class="chip">${escapeHTML(s)}</button>`).join("")}
        </div>
      `;
      empty.dataset.populated = "yes";
      attachChipHandlers();
    }
  }
}

function attachChipHandlers() {
  $$(".chip").forEach((chip) => {
    chip.onclick = () => {
      questionInput.value = chip.textContent.trim();
      askForm.requestSubmit();
    };
  });
}

// ─── ingest tabs ────────────────────────────────────────────────
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    $$(".tab-body").forEach((b) => b.classList.add("hidden"));
    $(`#tab-${tab.dataset.tab}`).classList.remove("hidden");
  });
});

// ─── ingest by path ─────────────────────────────────────────────
async function ingestPath(path) {
  const r = await fetchT("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths: [path] }),
  }, 120000);
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "Ingest failed");
  // total_chunks==0 with ingested_documents>0 means everything was already
  // indexed (idempotent re-ingest). That's fine — we surface it as a notice
  // rather than an error. Only fail when no documents were even read.
  if (!j.ingested_documents) {
    throw new Error("Ingest read 0 documents — check that the path exists and contains .txt/.md/.pdf files.");
  }
  return j;
}

$("#ingest-path-btn").addEventListener("click", async () => {
  const path = $("#ingest-path").value.trim();
  if (!path) return;
  const btn = $("#ingest-path-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>Ingesting…';
  try {
    const j = await ingestPath(path);
    showIngestResult(j, false);
  } catch (e) {
    showIngestResult({ error: e.message }, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Ingest";
    refreshStats();
  }
});

// ─── ingest by upload ───────────────────────────────────────────
const fileInput = $("#file-input");
const dropzone = $("#dropzone");
const fileList = $("#file-list");
const uploadBtn = $("#ingest-upload-btn");
let pendingFiles = [];

function renderFileList() {
  fileList.innerHTML = "";
  pendingFiles.forEach((f) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${escapeHTML(f.name)}</span><span>${(f.size / 1024).toFixed(1)} KB</span>`;
    fileList.appendChild(li);
  });
  uploadBtn.disabled = pendingFiles.length === 0;
}
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  pendingFiles = Array.from(e.target.files);
  renderFileList();
});
["dragover", "dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (evt === "dragover") dropzone.classList.add("dragover");
    if (evt === "dragleave" || evt === "drop") dropzone.classList.remove("dragover");
    if (evt === "drop") {
      pendingFiles = Array.from(e.dataTransfer.files);
      renderFileList();
    }
  });
});
uploadBtn.addEventListener("click", async () => {
  if (pendingFiles.length === 0) return;
  uploadBtn.disabled = true;
  uploadBtn.innerHTML = '<span class="spin"></span>Uploading…';
  try {
    const fd = new FormData();
    pendingFiles.forEach((f) => fd.append("files", f));
    const r = await fetch("/ingest/upload", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Upload failed");
    showIngestResult(j, false);
    pendingFiles = [];
    fileInput.value = "";
    renderFileList();
  } catch (e) {
    showIngestResult({ error: e.message }, true);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload & ingest";
    refreshStats();
  }
});

function showIngestResult(j, isErr) {
  const box = $("#ingest-result");
  box.classList.remove("hidden", "ok", "err");
  if (isErr) {
    box.classList.add("err");
    box.textContent = `Error: ${j.error}`;
    return;
  }
  box.classList.add("ok");
  if (j.total_chunks === 0) {
    box.innerHTML = `Already indexed — read <strong>${j.ingested_documents}</strong> document(s), <strong>0</strong> new chunks · index size <strong>${j.index_size}</strong>.`;
  } else {
    box.innerHTML = `Ingested <strong>${j.ingested_documents}</strong> document(s) · <strong>${j.total_chunks}</strong> new chunks · index size <strong>${j.index_size}</strong>.`;
  }
}

// ─── chat / ask ─────────────────────────────────────────────────
const chat = $("#chat");
const askForm = $("#ask-form");
const questionInput = $("#question");

// Initial chip handlers (in case the static HTML chips are still present).
attachChipHandlers();

function buildQueryBody(question, top_k, threshold) {
  const body = { question, top_k, score_threshold: threshold };
  if (activeSourceFilter.size > 0) {
    body.source_filter = [...activeSourceFilter];
  }
  return body;
}

async function doQuery(question, top_k, threshold) {
  const r = await fetchT("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildQueryBody(question, top_k, threshold)),
  }, 60000);
  return { status: r.status, body: await r.json() };
}

// ─── SSE streaming: fetch + manual event parser (EventSource is GET-only) ───
async function streamQuery(question, top_k, threshold, placeholder) {
  const resp = await fetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildQueryBody(question, top_k, threshold)),
  });
  if (!resp.ok || !resp.body) {
    const body = await resp.json().catch(() => ({}));
    return { status: resp.status, body };
  }

  placeholder.innerHTML = `
    <div class="answer-text" id="stream-answer"></div>
    <div class="confidence-row"><span class="hint"><span class="spin"></span>streaming…</span></div>
    <details class="sources hidden" id="stream-sources-block">
      <summary>📎 sources</summary>
      <div id="stream-sources-list"></div>
    </details>`;
  const answerEl = placeholder.querySelector("#stream-answer");
  const sourcesBlock = placeholder.querySelector("#stream-sources-block");
  const sourcesList = placeholder.querySelector("#stream-sources-list");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let acc = "";
  let firstSources = null;
  let doneEvent = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        let evt;
        try { evt = JSON.parse(line.slice(5).trim()); } catch { continue; }
        if (evt.type === "sources") {
          firstSources = evt.sources;
          sourcesList.innerHTML = (evt.sources || []).map(sourceCardHTML).join("");
          sourcesBlock.classList.remove("hidden");
          sourcesBlock.querySelector("summary").textContent =
            `📎 ${evt.sources.length} source${evt.sources.length === 1 ? "" : "s"}`;
          chat.scrollTop = chat.scrollHeight;
        } else if (evt.type === "token") {
          acc += evt.delta;
          answerEl.textContent = acc;
          chat.scrollTop = chat.scrollHeight;
        } else if (evt.type === "done") {
          doneEvent = evt;
        } else if (evt.type === "error") {
          throw new Error(evt.message || "stream error");
        }
      }
    }
  }

  return {
    status: 200,
    body: {
      question,
      answer: doneEvent && doneEvent.answer ? doneEvent.answer : acc,
      confident: !!(doneEvent && !doneEvent.rejected),
      confidence_score: (doneEvent && doneEvent.confidence_score) || 0,
      confidence_explanation: doneEvent && doneEvent.confidence_explanation,
      rejected: !!(doneEvent && doneEvent.rejected),
      rejection_reason: doneEvent && doneEvent.rejection_reason,
      sources: firstSources || [],
      metadata: { streamed: true, top_k },
    },
  };
}

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  const empty = chat.querySelector(".empty");
  if (empty) empty.remove();

  appendMessage("q", question);
  const placeholder = appendMessage("a", '<span class="spin"></span>Thinking…', true);
  questionInput.value = "";

  const top_k = parseInt($("#top-k").value, 10) || 4;
  const thresholdRaw = parseFloat($("#threshold").value);
  const threshold = isNaN(thresholdRaw) ? null : thresholdRaw;
  const useStream = $("#stream-toggle").checked;

  try {
    const { status, body } = useStream
      ? await streamQuery(question, top_k, threshold, placeholder)
      : await doQuery(question, top_k, threshold);

    if (status === 409) {
      placeholder.innerHTML = `
        <div style="color:var(--danger)">${escapeHTML(body.detail || "Index empty")}</div>
        <div class="recovery-action">
          <button class="primary" id="recover-btn">🚀 Initialize with example docs</button>
        </div>`;
      document.getElementById("recover-btn").onclick = async () => {
        const b = document.getElementById("recover-btn");
        b.disabled = true;
        b.innerHTML = '<span class="spin"></span>Ingesting…';
        try {
          await ingestPath("data/docs");
          await refreshStats();
          b.innerHTML = "✓ Ready — ask again";
        } catch (e) {
          b.innerHTML = `Failed: ${e.message}`;
          b.disabled = false;
        }
      };
      return;
    }

    if (status >= 400) throw new Error(body.detail || "Query failed");
    placeholder.replaceWith(renderAnswer(body));
  } catch (err) {
    placeholder.innerHTML = `<span style="color:var(--danger)">Error: ${escapeHTML(err.message)}</span>`;
  } finally {
    chat.scrollTop = chat.scrollHeight;
    refreshStats();
  }
});

function appendMessage(role, html, isHTML = false) {
  const div = document.createElement("div");
  div.className = role === "q" ? "msg-q" : "msg-a";
  if (isHTML) div.innerHTML = html;
  else div.textContent = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function sourceCardHTML(s) {
  return `
    <div class="source">
      <div class="source-head">
        <strong>[#${s.rank}] ${escapeHTML(basename(s.source))}</strong>
        <span>score ${Number(s.score).toFixed(3)}</span>
      </div>
      <div class="source-snippet">${escapeHTML(s.snippet)}</div>
    </div>`;
}

function retrievalBadgesHTML(meta) {
  if (!meta) return "";
  const badges = [];
  const r = meta.retrieval || {};
  if (r.mode === "hybrid") badges.push(`<span class="r-badge b-hybrid">hybrid</span>`);
  else if (r.mode === "dense") badges.push(`<span class="r-badge">dense</span>`);
  if (r.mmr_applied) badges.push(`<span class="r-badge b-mmr">mmr</span>`);
  if (r.rerank_applied) badges.push(`<span class="r-badge b-rerank">rerank</span>`);
  if (r.source_filter && r.source_filter.length) {
    badges.push(`<span class="r-badge b-filter">filter:${escapeHTML(r.source_filter.join(","))}</span>`);
  }
  if (meta.served_from_cache) badges.push(`<span class="r-badge b-cache">cached</span>`);
  if (meta.streamed) badges.push(`<span class="r-badge">streamed</span>`);
  return badges.length ? `<span class="retrieval-badges">${badges.join("")}</span>` : "";
}

function renderAnswer(j) {
  const wrap = document.createElement("div");
  wrap.className = "msg-a";

  const conf = j.confidence_score || 0;
  let confClass = "conf-low", confLabel = "abstained";
  if (j.confident && conf >= 0.55) { confClass = "conf-high"; confLabel = "high confidence"; }
  else if (j.confident) { confClass = "conf-mid"; confLabel = "moderate confidence"; }

  const sourcesHTML = (j.sources || []).map(sourceCardHTML).join("");
  const meta = j.metadata || {};
  const backend = meta.embedding_backend || "";
  const topK = meta.top_k || "";

  wrap.innerHTML = `
    <div class="answer-text">${escapeHTML(j.answer)}</div>
    <div class="confidence-row">
      <span class="conf-badge ${confClass}">● ${confLabel} · ${conf.toFixed(2)}</span>
      <span>${escapeHTML(backend)}${topK ? " · top-" + topK : ""}${retrievalBadgesHTML(meta)}</span>
    </div>
    ${j.sources && j.sources.length ? `
      <details class="sources">
        <summary>📎 ${j.sources.length} source${j.sources.length === 1 ? "" : "s"}</summary>
        ${sourcesHTML}
      </details>` : ""}
  `;
  return wrap;
}

// ─── evaluate ───────────────────────────────────────────────────
$("#evaluate-btn").addEventListener("click", async () => {
  const btn = $("#evaluate-btn");
  const summary = $("#eval-summary");
  const canvas = $("#chart-eval");
  const path = $("#qa-path").value.trim() || "data/eval/qa_pairs.json";
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>Running…';
  summary.classList.remove("hidden", "ok", "err");
  summary.textContent = "";
  try {
    const r = await fetch("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ qa_path: path }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Evaluation failed");
    summary.classList.add("ok");
    summary.innerHTML = `Evaluated <strong>${j.num_questions}</strong> questions · saved to <code>${escapeHTML(j.results_path)}</code>`;
    paintEvalChart(j.aggregate);
    canvas.classList.remove("hidden");
  } catch (e) {
    summary.classList.add("err");
    summary.textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run";
  }
});

function paintEvalChart(agg) {
  const labels = ["Exact match", "Semantic sim", "Retrieval@k", "Answered"];
  const data = [
    agg.exact_match || 0,
    agg.semantic_similarity || 0,
    agg.retrieval_accuracy || 0,
    agg.answered || 0,
  ].map((x) => +(x * 100).toFixed(1));
  const colors = [PRIMARY, ACCENT, SUCCESS, WARN];
  const c = themeColors();
  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    animations: { colors: false, x: false, y: false },
    plugins: { tooltip: { mode: "index" } },
    scales: {
      x: { grid: { display: false }, ticks: { color: c.muted } },
      y: { beginAtZero: true, max: 100, ticks: { callback: (v) => v + "%", color: c.muted }, grid: { color: c.grid } },
    },
  };

  if (!evalChart) {
    evalChart = new Chart($("#chart-eval"), {
      type: "bar",
      data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 8 }] },
      options: opts,
    });
  } else {
    evalChart.data.labels = labels;
    evalChart.data.datasets[0].data = data;
    evalChart.update("none");
  }
}

// ─── boot ───────────────────────────────────────────────────────
// Refresh cadence: 30s is plenty for a dashboard. The previous 8s loop
// caused visible flicker/redraw on the charts every few seconds, which
// made the page feel "hectic" and pushed cards below it around.
applyChartTheme();
refreshStats();
setInterval(refreshStats, 30000);
