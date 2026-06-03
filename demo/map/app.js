/* ═══════════════════════════════════════════════════════════════════════════
   Palimpsest — interactive change-detection demo
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = "/api/v1";

// ── Demo presets ──────────────────────────────────────────────────────────────
// Each bbox is kept ≤ ~8 km × 8 km so EE export + inference finishes in < 3 min.
const DEMOS = [
  {
    label: "🇱🇧 Beirut — port explosion (Aug 2020)",
    bbox:   [35.503, 33.888, 35.536, 33.918],   // 3.3 × 3.3 km ✓
    before: "2020-07-01", after: "2020-09-15",
    name: "Beirut Port Explosion", cloud: 20,
    zoom: [33.903, 35.519, 14],
  },
  {
    label: "🇦🇪 Dubai — Creek Harbour construction",
    bbox:   [55.318, 25.178, 55.378, 25.228],   // 5.8 × 5.6 km ✓
    before: "2017-01-01", after: "2022-01-01",
    name: "Dubai Creek Harbour", cloud: 5,
    zoom: [25.203, 55.348, 13],
  },
  {
    label: "🇸🇦 NEOM — The Line construction site",
    bbox:   [36.53, 28.06, 36.62, 28.13],       // 8.7 × 7.8 km ✓
    before: "2021-01-01", after: "2024-01-01",
    name: "NEOM The Line", cloud: 5,
    zoom: [28.095, 36.575, 13],
  },
  {
    label: "🇸🇦 Riyadh — northern urban expansion",
    bbox:   [46.68, 24.86, 46.77, 24.94],       // 8.7 × 8.9 km ✓
    before: "2017-01-01", after: "2023-01-01",
    name: "Riyadh North Expansion", cloud: 5,
    zoom: [24.90, 46.725, 13],
  },
  {
    label: "🇨🇳 Shenzhen — Qianhai Bay reclamation",
    bbox:   [113.88, 22.50, 113.97, 22.57],     // 8.7 × 7.8 km ✓
    before: "2016-01-01", after: "2022-01-01",
    name: "Shenzhen Qianhai", cloud: 20,
    zoom: [22.535, 113.925, 14],
  },
  {
    label: "🇺🇸 Las Vegas — Henderson development",
    bbox:   [-115.07, 36.00, -114.98, 36.07],   // 8.4 × 7.8 km ✓
    before: "2016-01-01", after: "2023-01-01",
    name: "Las Vegas Henderson", cloud: 5,
    zoom: [36.035, -115.025, 13],
  },
  {
    label: "🇧🇷 Amazon — Pará deforestation",
    bbox:   [-55.05, -4.83, -54.96, -4.74],     // 8.7 × 10 km ✓
    before: "2018-06-01", after: "2022-06-01",
    name: "Amazon Deforestation", cloud: 30,
    zoom: [-4.785, -55.005, 13],
  },
  {
    label: "🇺🇦 Kakhovka dam — flood zone (Jun 2023)",
    bbox:   [33.46, 47.00, 33.57, 47.08],       // 7.9 × 8.9 km ✓
    before: "2023-04-01", after: "2023-08-01",
    name: "Kakhovka Flood", cloud: 30,
    zoom: [47.04, 33.515, 13],
  },
  {
    label: "🇪🇬 New Alamein — desert city construction",
    bbox:   [28.94, 30.78, 29.03, 30.86],       // 8.7 × 8.9 km ✓
    before: "2018-01-01", after: "2023-01-01",
    name: "New Alamein City", cloud: 5,
    zoom: [30.82, 28.985, 13],
  },
  {
    label: "🇸🇦 NEOM — Sindalah island",
    bbox:   [37.20, 27.90, 37.30, 27.97],       // 9.6 × 7.8 km ✓
    before: "2022-01-01", after: "2024-06-01",
    name: "Sindalah Island", cloud: 5,
    zoom: [27.935, 37.250, 13],
  },
];

// ── Map setup ─────────────────────────────────────────────────────────────────
const map = L.map("map").setView([20, 10], 3);

const street = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© <a href='https://openstreetmap.org'>OpenStreetMap</a>",
  maxZoom: 19,
}).addTo(map);

const satellite = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { attribution: "© Esri", maxZoom: 19 }
);

L.control.layers({ Street: street, Satellite: satellite }).addTo(map);

// Keep Leaflet in sync whenever the map container resizes (e.g. sidebar grows)
new ResizeObserver(() => map.invalidateSize()).observe(document.getElementById("map"));

// ── State ─────────────────────────────────────────────────────────────────────
let bboxRect      = null;
let drawStart     = null;
let changeLayer   = null;
let elapsedTimer  = null;
let lastDetection = null;   // { detection_id, polygon_count }

// ── Populate demo dropdown ────────────────────────────────────────────────────
const demoSelect = document.getElementById("demo-select");
DEMOS.forEach((d, i) => {
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = d.label;
  demoSelect.appendChild(opt);
});

demoSelect.addEventListener("change", () => {
  const idx = parseInt(demoSelect.value, 10);
  if (isNaN(idx)) return;
  const d = DEMOS[idx];
  applyBbox(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]);
  document.getElementById("date-before").value = d.before;
  document.getElementById("date-after").value  = d.after;
  document.getElementById("area-name").value   = d.name;
  document.getElementById("cloud-pct").value   = d.cloud ?? 15;
  document.getElementById("cloud-pct-val").textContent = (d.cloud ?? 15) + " %";
  if (d.zoom) map.setView([d.zoom[0], d.zoom[1]], d.zoom[2]);
  hideError();
  hideResults();
});

// ── Apply coords button ───────────────────────────────────────────────────────
document.getElementById("btn-apply-coords").addEventListener("click", applyFromInputs);
["c-west","c-east","c-south","c-north"].forEach(id => {
  document.getElementById(id).addEventListener("keydown", e => {
    if (e.key === "Enter") applyFromInputs();
  });
});

function applyFromInputs() {
  const w = parseFloat(document.getElementById("c-west").value);
  const e = parseFloat(document.getElementById("c-east").value);
  const s = parseFloat(document.getElementById("c-south").value);
  const n = parseFloat(document.getElementById("c-north").value);
  if ([w,e,s,n].some(isNaN)) { showError("Enter all four coordinates first."); return; }
  if (w >= e) { showError("W (lon min) must be less than E (lon max)."); return; }
  if (s >= n) { showError("S (lat min) must be less than N (lat max)."); return; }
  hideError();
  applyBbox(w, s, e, n);
}

// ── Core bbox helper ──────────────────────────────────────────────────────────
function applyBbox(w, s, e, n) {
  // Draw rectangle on map
  if (bboxRect) { bboxRect.remove(); bboxRect = null; }
  const bounds = L.latLngBounds([s, w], [n, e]);
  bboxRect = L.rectangle(bounds, { color: "#4a90d9", weight: 2, fillOpacity: .1 }).addTo(map);
  map.fitBounds(bounds.pad(0.15));
  // Sync coordinate inputs
  document.getElementById("c-west").value  = w;
  document.getElementById("c-east").value  = e;
  document.getElementById("c-south").value = s;
  document.getElementById("c-north").value = n;
}

// ── Cloud % slider live label ─────────────────────────────────────────────────
const cloudSlider = document.getElementById("cloud-pct");
const cloudVal    = document.getElementById("cloud-pct-val");
cloudSlider.addEventListener("input", () => { cloudVal.textContent = cloudSlider.value + " %"; });

// ── Bbox draw (Shift+drag) ────────────────────────────────────────────────────
map.on("mousedown", (e) => {
  if (!e.originalEvent.shiftKey) return;
  drawStart = e.latlng;
  map.dragging.disable();
  if (bboxRect) { bboxRect.remove(); bboxRect = null; }
});

map.on("mousemove", (e) => {
  if (!drawStart) return;
  const bounds = L.latLngBounds(drawStart, e.latlng);
  if (bboxRect) bboxRect.remove();
  bboxRect = L.rectangle(bounds, { color: "#4a90d9", weight: 2, fillOpacity: .1 }).addTo(map);
});

map.on("mouseup", (e) => {
  if (!drawStart) return;
  map.dragging.enable();
  const bounds = L.latLngBounds(drawStart, e.latlng);
  drawStart = null;
  if (bboxRect) bboxRect.remove();
  bboxRect = L.rectangle(bounds, { color: "#4a90d9", weight: 2, fillOpacity: .1 }).addTo(map);
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  // Sync inputs (no fitBounds — user just drew it)
  document.getElementById("c-west").value  = sw.lng.toFixed(4);
  document.getElementById("c-east").value  = ne.lng.toFixed(4);
  document.getElementById("c-south").value = sw.lat.toFixed(4);
  document.getElementById("c-north").value = ne.lat.toFixed(4);
  demoSelect.value = "";
  hideError();
});

// ── Detect button ─────────────────────────────────────────────────────────────
document.getElementById("btn-detect").addEventListener("click", async () => {
  if (!bboxRect) {
    showError("Select a demo location or draw a bounding box (Shift+drag) first.");
    return;
  }

  const bounds     = bboxRect.getBounds();
  const sw         = bounds.getSouthWest();
  const ne         = bounds.getNorthEast();
  const bbox       = [sw.lng, sw.lat, ne.lng, ne.lat];

  // Warn if the bbox is too large (> ~15 km side) — EE export would time out.
  const dLon = Math.abs(ne.lng - sw.lng);
  const dLat = Math.abs(ne.lat - sw.lat);
  const kmW  = dLon * 111.32 * Math.cos((sw.lat + ne.lat) / 2 * Math.PI / 180);
  const kmH  = dLat * 110.57;
  if (kmW > 15 || kmH > 15) {
    showError(
      `Area too large (≈${kmW.toFixed(0)} × ${kmH.toFixed(0)} km). ` +
      `Please draw a smaller box (max ~10 km per side) or pick a demo location.`
    );
    return;
  }

  const dateBefore = document.getElementById("date-before").value;
  const dateAfter  = document.getElementById("date-after").value;
  if (!dateBefore || !dateAfter) { showError("Please set both dates."); return; }
  if (dateBefore >= dateAfter)   { showError("'Before' date must be earlier than 'After'."); return; }

  const maxCloud = parseFloat(cloudSlider.value);
  const areaName = document.getElementById("area-name").value.trim() || undefined;

  hideError();
  hideResults();
  if (changeLayer) { changeLayer.remove(); changeLayer = null; }

  const btn = document.getElementById("btn-detect");
  btn.disabled = true;
  startProgress();

  try {
    const resp = await fetch(`${API_BASE}/detect`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox, date_before: dateBefore, date_after: dateAfter,
                             max_cloud_pct: maxCloud, area_name: areaName }),
    });

    stopProgress();

    let data;
    const ct = resp.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      data = await resp.json();
    } else {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text.slice(0, 120)}`);
    }
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);

    lastDetection = {
      detection_id:  data.detection_id,
      polygon_count: data.polygons?.length ?? 0,
    };
    showResults(data);

    if (data.polygons?.length) {
      changeLayer = L.geoJSON(
        { type: "FeatureCollection", features: data.polygons },
        {
          style: { color: "#e74c3c", weight: 1.5, fillColor: "#e74c3c", fillOpacity: 0.45 },
          onEachFeature: (feature, layer) => {
            const a = feature.properties?.area_m2;
            if (a != null) layer.bindTooltip(formatArea(a), { sticky: true });
          },
        }
      ).addTo(map);
    }

    // Always zoom to the AOI bounding box — changeLayer.getBounds() can
    // include outlier polygons that span the globe, causing fitBounds
    // to zoom to world view.  The bboxRect is always correct.
    setTimeout(() => {
      map.invalidateSize();
      if (bboxRect) {
        map.fitBounds(bboxRect.getBounds().pad(0.15));
      }
    }, 200);

  } catch (err) {
    stopProgress();
    showError(err.message || "Detection failed.");
    console.error(err);
  } finally {
    btn.disabled = false;
  }
});

// ── Progress ──────────────────────────────────────────────────────────────────
const EXPECTED_DURATION_S = 75;

function startProgress() {
  const bar  = document.getElementById("status-bar");
  const fill = document.getElementById("progress-fill");
  const txt  = document.getElementById("status-text");
  const el   = document.getElementById("elapsed");
  bar.classList.remove("hidden");
  fill.style.width = "0%";
  txt.textContent  = "Searching Sentinel-2 scenes via Earth Engine…";
  el.textContent   = "";

  let secs = 0;
  elapsedTimer = setInterval(() => {
    secs++;
    el.textContent = `${secs}s elapsed`;
    fill.style.width = Math.min(95, Math.round((secs / EXPECTED_DURATION_S) * 100)) + "%";
    if (secs === 8)  txt.textContent = "Exporting GeoTIFFs to Cloud Storage…";
    if (secs === 30) txt.textContent = "Running ChangeFormer model…";
    if (secs === 55) txt.textContent = "Saving results to BigQuery…";
  }, 1000);
}

function stopProgress() {
  clearInterval(elapsedTimer);
  elapsedTimer = null;
  document.getElementById("progress-fill").style.width = "100%";
  document.getElementById("status-text").textContent = "";
  setTimeout(() => {
    document.getElementById("status-bar").classList.add("hidden");
    document.getElementById("progress-fill").style.width = "0%";
  }, 600);
}

// ── Results ───────────────────────────────────────────────────────────────────
function showResults(data) {
  document.getElementById("r-before").textContent = data.date_before_actual ?? "—";
  document.getElementById("r-after").textContent  = data.date_after_actual  ?? "—";
  document.getElementById("r-area").textContent   = formatArea(data.change_area_m2);
  document.getElementById("r-pct").textContent    = data.change_pct != null
    ? data.change_pct.toFixed(2) + " %" : "—";
  document.getElementById("r-poly").textContent   = data.polygons?.length ?? 0;
  document.getElementById("r-time").textContent   = data.processing_time_s != null
    ? data.processing_time_s + " s" : "—";

  const idEl = document.getElementById("r-id");
  idEl.textContent = data.detection_id ? data.detection_id.slice(0, 16) + "…" : "—";
  idEl.title = data.detection_id ?? "";
  idEl.onclick = () => {
    navigator.clipboard?.writeText(data.detection_id);
    idEl.textContent = "Copied!";
    setTimeout(() => { idEl.textContent = data.detection_id.slice(0, 16) + "…"; }, 1200);
  };

  // Show analyse button only when we have an ID (required by /analyse endpoint)
  const btnAnalyse = document.getElementById("btn-analyse");
  if (data.detection_id) {
    btnAnalyse.classList.remove("hidden");
  } else {
    btnAnalyse.classList.add("hidden");
  }

  document.getElementById("results-card").classList.remove("hidden");
  hideAnalysis();
}

function hideResults() {
  document.getElementById("results-card").classList.add("hidden");
  hideAnalysis();
}

// ── AI Analysis ───────────────────────────────────────────────────────────────
document.getElementById("btn-analyse").addEventListener("click", async () => {
  if (!lastDetection?.detection_id) return;
  await runAnalysis(lastDetection.detection_id, lastDetection.polygon_count);
});

async function runAnalysis(detectionId, polygonCount) {
  const btn = document.getElementById("btn-analyse");
  btn.disabled = true;
  btn.textContent = "⏳ Analysing…";
  hideAnalysis();
  showAnalysisProgress();

  try {
    const resp = await fetch(`${API_BASE}/detect/${detectionId}/analyse`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ polygon_count: polygonCount || null }),
    });

    stopAnalysisProgress();

    let data;
    const ct = resp.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      data = await resp.json();
    } else {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text.slice(0, 120)}`);
    }

    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);

    renderAnalysis(data);

  } catch (err) {
    stopAnalysisProgress();
    showError(err.message || "Analysis failed.");
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍 Analyse with AI";
  }
}

function renderAnalysis(data) {
  document.getElementById("a-location").textContent = data.location || "—";
  document.getElementById("a-summary").textContent  = data.summary  || "—";
  document.getElementById("a-analysis").textContent = data.analysis || "—";

  // Likely causes — bullet list
  const causesEl = document.getElementById("a-causes");
  causesEl.innerHTML = "";
  if (data.likely_causes?.length) {
    const ul = document.createElement("ul");
    data.likely_causes.forEach(c => {
      const li = document.createElement("li");
      li.textContent = c;
      ul.appendChild(li);
    });
    causesEl.appendChild(ul);
  }

  // Confidence badge
  const conf = (data.confidence || "MEDIUM").toUpperCase();
  const confEl = document.getElementById("a-confidence");
  confEl.textContent = conf;
  confEl.className = "conf-badge conf-" + conf.toLowerCase();

  // Context sources (collapsible)
  const ctxWrap = document.getElementById("a-context-wrap");
  const ctxEl   = document.getElementById("a-context");
  if (data.context_used?.length) {
    ctxEl.innerHTML = "";
    data.context_used.forEach(snippet => {
      const p = document.createElement("p");
      p.className = "a-context-item";
      p.textContent = snippet;
      ctxEl.appendChild(p);
    });
    ctxWrap.style.display = "";
  } else {
    ctxWrap.style.display = "none";
  }

  // Generated timestamp
  if (data.generated_at) {
    const ts = new Date(data.generated_at);
    const tsEl = document.createElement("div");
    tsEl.className = "a-timestamp";
    tsEl.textContent = "Generated " + ts.toUTCString().replace("GMT", "UTC");
    document.getElementById("a-context-wrap").after(tsEl);
  }

  document.getElementById("analysis-card").classList.remove("hidden");

  // Re-zoom to AOI after analysis card expands the sidebar.
  setTimeout(() => {
    map.invalidateSize();
    if (bboxRect) {
      map.fitBounds(bboxRect.getBounds().pad(0.15));
    }
  }, 200);
}

function hideAnalysis() {
  document.getElementById("analysis-card").classList.add("hidden");
  document.getElementById("analysis-status").classList.add("hidden");
  // Remove any lingering timestamp elements
  document.querySelectorAll(".a-timestamp").forEach(el => el.remove());
}

function showAnalysisProgress() {
  const el = document.getElementById("analysis-status");
  el.classList.remove("hidden");
  document.getElementById("analysis-progress-fill").style.width = "0%";
  document.getElementById("analysis-status-text").textContent   = "Querying Gemini…";

  let pct = 0;
  el._timer = setInterval(() => {
    pct = Math.min(90, pct + (pct < 30 ? 3 : pct < 70 ? 1.5 : 0.5));
    document.getElementById("analysis-progress-fill").style.width = pct + "%";
    if (pct > 20) document.getElementById("analysis-status-text").textContent = "Reverse-geocoding centroid…";
    if (pct > 45) document.getElementById("analysis-status-text").textContent = "Searching Wikipedia…";
    if (pct > 70) document.getElementById("analysis-status-text").textContent = "Generating structured analysis…";
  }, 500);
}

function stopAnalysisProgress() {
  const el = document.getElementById("analysis-status");
  clearInterval(el._timer);
  document.getElementById("analysis-progress-fill").style.width = "100%";
  setTimeout(() => el.classList.add("hidden"), 500);
}

// ── Error helpers ─────────────────────────────────────────────────────────────
function showError(msg) {
  const el = document.getElementById("error-box");
  el.textContent = "⚠ " + msg;
  el.classList.remove("hidden");
}
function hideError() {
  document.getElementById("error-box").classList.add("hidden");
}

// ── Formatting ────────────────────────────────────────────────────────────────
function formatArea(m2) {
  if (m2 == null) return "—";
  if (m2 >= 1_000_000) return (m2 / 1_000_000).toFixed(2) + " km²";
  if (m2 >= 10_000)    return (m2 / 10_000).toFixed(1) + " ha";
  return Math.round(m2) + " m²";
}
