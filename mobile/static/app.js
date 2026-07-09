/* ══════════════════════════════════════════════════════════════════
   Composte IA — annotation mobile
   Éditeur de bounding boxes 100% tactile (Pointer Events), sans dépendance.

   Systèmes de coordonnées :
   - image  : pixels de l'image ORIGINALE (ce qui part au serveur)
   - écran  : pixels CSS du canvas ; écran = image * (baseScale*zoom) + pan
   Gestes  : 1 doigt = dessiner / sélectionner / déplacer / redimensionner
             2 doigts = pinch-zoom + pan · double-tap = recadrer
   ══════════════════════════════════════════════════════════════════ */
"use strict";

const $ = (id) => document.getElementById(id);

const HANDLE_HIT_PX = 26;   // rayon de capture des poignées (px écran)
const TAP_SLOP_PX   = 8;    // en-dessous : c'est un tap, pas un drag
const MIN_BOX_IMG   = 5;    // taille mini d'une bbox (px image)
const HISTORY_MAX   = 60;

const state = {
  config: null,                       // {classes, colors, conf_default}
  sessionId: "mob_" + new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14),
  conf: 0.4,
  queue: [],                          // [{file, name, fromDataset}]
  queueTotal: 0,
  current: null,                      // {file, name, fromDataset, img, w, h, url}
  imageToken: 0,                      // invalide les réponses réseau des images précédentes
  boxes: [],                          // [{x, y, w, h, label_id}] en px image
  selected: -1,
  currentClass: 0,
  history: [],
  future: [],
  highlight: false,
  startTime: 0,
  galleryFilter: "all",
  saving: false,
};

const view = { baseScale: 1, zoom: 1, panX: 0, panY: 0 };

const canvas = $("canvas");
const wrap = $("canvas-wrap");
const ctx = canvas.getContext("2d");
let dpr = window.devicePixelRatio || 1;

/* ══════════════════ NAVIGATION ENTRE VUES ══════════════════ */

function showView(name) {
  for (const v of document.querySelectorAll(".view")) v.classList.remove("active");
  $("view-" + name).classList.add("active");
  if (name === "editor") resizeCanvas();
}

/* ══════════════════ OUTILS UI ══════════════════ */

let toastTimer = null;
function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

function fmtDuration(sec) {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/* ══════════════════ CONVERSIONS DE COORDONNÉES ══════════════════ */

const scaleTotal = () => view.baseScale * view.zoom;
const imgToScreen = (x, y) => ({ x: x * scaleTotal() + view.panX, y: y * scaleTotal() + view.panY });
const screenToImg = (x, y) => ({ x: (x - view.panX) / scaleTotal(), y: (y - view.panY) / scaleTotal() });

function fitView() {
  if (!state.current) return;
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  view.baseScale = Math.min(cw / state.current.w, ch / state.current.h);
  view.zoom = 1;
  view.panX = (cw - state.current.w * view.baseScale) / 2;
  view.panY = (ch - state.current.h * view.baseScale) / 2;
}

function resizeCanvas() {
  dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(wrap.clientWidth * dpr);
  canvas.height = Math.round(wrap.clientHeight * dpr);
  fitView();
  draw();
}
new ResizeObserver(resizeCanvas).observe(wrap);

/* ══════════════════ RENDU ══════════════════ */

let draftBox = null; // bbox en cours de dessin {x0,y0,x1,y1} (px image)

function classColor(id) {
  const c = state.config;
  return c ? c.colors[c.classes[id]] || "#ff5555" : "#ff5555";
}

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.current || !state.current.img) return;

  const s = scaleTotal();
  ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * view.panX, dpr * view.panY);
  ctx.imageSmoothingEnabled = s < 1;
  ctx.drawImage(state.current.img, 0, 0);

  const lw = 2.4 / s;
  ctx.font = `${13 / s}px system-ui, sans-serif`;

  state.boxes.forEach((b, i) => {
    const color = classColor(b.label_id);
    ctx.lineWidth = i === state.selected ? lw * 1.5 : lw;
    ctx.strokeStyle = color;
    if (state.highlight) {
      ctx.fillStyle = color + "48";
      ctx.fillRect(b.x, b.y, b.w, b.h);
    }
    ctx.strokeRect(b.x, b.y, b.w, b.h);

    // étiquette de classe sur fond coloré
    const label = state.config ? state.config.classes[b.label_id] : "?";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(b.x, b.y - 17 / s, tw + 10 / s, 17 / s);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, b.x + 5 / s, b.y - 4.5 / s);
  });

  // bbox en cours de dessin
  if (draftBox) {
    const color = classColor(state.currentClass);
    ctx.lineWidth = lw;
    ctx.strokeStyle = color;
    ctx.fillStyle = color + "30";
    const r = normRect(draftBox);
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeRect(r.x, r.y, r.w, r.h);
  }

  // poignées de la bbox sélectionnée
  if (state.selected >= 0 && state.boxes[state.selected]) {
    const b = state.boxes[state.selected];
    const hs = 11 / s; // taille visuelle des poignées
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = classColor(b.label_id);
    ctx.lineWidth = lw;
    for (const h of handlePositions(b)) {
      ctx.fillRect(h.x - hs / 2, h.y - hs / 2, hs, hs);
      ctx.strokeRect(h.x - hs / 2, h.y - hs / 2, hs, hs);
    }
  }

  updateBadges();
}

function handlePositions(b) {
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
  return [
    { id: "nw", x: b.x, y: b.y },        { id: "n", x: cx, y: b.y },        { id: "ne", x: b.x + b.w, y: b.y },
    { id: "w",  x: b.x, y: cy },                                            { id: "e",  x: b.x + b.w, y: cy },
    { id: "sw", x: b.x, y: b.y + b.h },  { id: "s", x: cx, y: b.y + b.h },  { id: "se", x: b.x + b.w, y: b.y + b.h },
  ];
}

function normRect(d) {
  return {
    x: Math.min(d.x0, d.x1),
    y: Math.min(d.y0, d.y1),
    w: Math.abs(d.x1 - d.x0),
    h: Math.abs(d.y1 - d.y0),
  };
}

function clampBox(b) {
  const W = state.current.w, H = state.current.h;
  if (b.w < 0) { b.x += b.w; b.w = -b.w; }
  if (b.h < 0) { b.y += b.h; b.h = -b.h; }
  b.w += Math.min(0, b.x); b.x = Math.max(0, b.x);
  b.h += Math.min(0, b.y); b.y = Math.max(0, b.y);
  b.w = Math.max(MIN_BOX_IMG, Math.min(b.w, W - b.x));
  b.h = Math.max(MIN_BOX_IMG, Math.min(b.h, H - b.y));
  b.x = Math.min(Math.max(0, b.x), Math.max(0, W - b.w));
  b.y = Math.min(Math.max(0, b.y), Math.max(0, H - b.h));
  return b;
}

function updateBadges() {
  $("editor-count").textContent = `${state.boxes.length} bbox${state.boxes.length !== 1 ? "s" : ""}`;
  $("btn-undo").disabled = state.history.length === 0;
  $("btn-redo").disabled = state.future.length === 0;
  $("btn-delete").disabled = state.selected < 0;
}

/* ══════════════════ HISTORIQUE ══════════════════ */

const snapshot = () => state.boxes.map((b) => ({ ...b }));

function pushHistory() {
  state.history.push(snapshot());
  if (state.history.length > HISTORY_MAX) state.history.shift();
  state.future = [];
}

function undo() {
  if (!state.history.length) return;
  state.future.push(snapshot());
  state.boxes = state.history.pop();
  state.selected = -1;
  draw();
}

function redo() {
  if (!state.future.length) return;
  state.history.push(snapshot());
  state.boxes = state.future.pop();
  state.selected = -1;
  draw();
}

function deleteSelected() {
  if (state.selected < 0) return;
  pushHistory();
  state.boxes.splice(state.selected, 1);
  state.selected = -1;
  draw();
}

/* ══════════════════ GESTES (Pointer Events) ══════════════════ */

const pointers = new Map(); // pointerId → {x, y} px CSS canvas
let gesture = null;         // {type, ...} — voir chaque cas
let lastTap = { t: 0, x: 0, y: 0 };

function pointerPos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function hitHandle(p) {
  if (state.selected < 0) return null;
  const b = state.boxes[state.selected];
  for (const h of handlePositions(b)) {
    const sp = imgToScreen(h.x, h.y);
    if (Math.hypot(sp.x - p.x, sp.y - p.y) <= HANDLE_HIT_PX) return h.id;
  }
  return null;
}

function hitBox(p) {
  const ip = screenToImg(p.x, p.y);
  // du dernier dessiné au premier → priorité au plus récent (dessus)
  for (let i = state.boxes.length - 1; i >= 0; i--) {
    const b = state.boxes[i];
    if (ip.x >= b.x && ip.x <= b.x + b.w && ip.y >= b.y && ip.y <= b.y + b.h) return i;
  }
  return -1;
}

canvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  const p = pointerPos(e);
  pointers.set(e.pointerId, p);

  if (pointers.size === 2) {
    // 2e doigt : on bascule en pinch, en annulant le geste 1 doigt en cours
    if (gesture && (gesture.type === "move" || gesture.type === "resize")) {
      state.boxes = gesture.before; // revert
    }
    draftBox = null;
    const [p1, p2] = [...pointers.values()];
    gesture = {
      type: "pinch",
      dist: Math.hypot(p2.x - p1.x, p2.y - p1.y),
      mid: { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 },
      zoom: view.zoom,
      panX: view.panX,
      panY: view.panY,
      baseScale: view.baseScale,
    };
    draw();
    return;
  }
  if (pointers.size !== 1 || gesture) return;

  const handle = hitHandle(p);
  if (handle) {
    gesture = { type: "resize", handle, before: snapshot(), start: screenToImg(p.x, p.y), moved: false };
    return;
  }
  const idx = hitBox(p);
  if (idx >= 0) {
    if (idx !== state.selected) {
      state.selected = idx;
      setCurrentClass(state.boxes[idx].label_id, false);
      draw();
    }
    gesture = { type: "maybe-move", idx, before: snapshot(), startScreen: p, start: screenToImg(p.x, p.y) };
    return;
  }
  gesture = { type: "maybe-draw", startScreen: p, start: screenToImg(p.x, p.y) };
});

canvas.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  const p = pointerPos(e);
  pointers.set(e.pointerId, p);
  if (!gesture) return;

  if (gesture.type === "pinch" && pointers.size === 2) {
    const [p1, p2] = [...pointers.values()];
    const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
    const newZoom = Math.min(Math.max(gesture.zoom * dist / gesture.dist, 0.5), 14);
    // point image sous le centre initial du geste → reste sous le centre actuel
    const s0 = gesture.baseScale * gesture.zoom;
    const origin = { x: (gesture.mid.x - gesture.panX) / s0, y: (gesture.mid.y - gesture.panY) / s0 };
    const s1 = gesture.baseScale * newZoom;
    view.zoom = newZoom;
    view.panX = mid.x - origin.x * s1;
    view.panY = mid.y - origin.y * s1;
    draw();
    return;
  }

  const ip = screenToImg(p.x, p.y);

  if (gesture.type === "maybe-draw") {
    if (Math.hypot(p.x - gesture.startScreen.x, p.y - gesture.startScreen.y) > TAP_SLOP_PX) {
      gesture = { type: "draw" };
      draftBox = { x0: ip.x, y0: ip.y, x1: ip.x, y1: ip.y };
    }
    return;
  }
  if (gesture.type === "draw" && draftBox) {
    draftBox.x1 = ip.x;
    draftBox.y1 = ip.y;
    draw();
    return;
  }
  if (gesture.type === "maybe-move") {
    if (Math.hypot(p.x - gesture.startScreen.x, p.y - gesture.startScreen.y) > TAP_SLOP_PX) {
      gesture.type = "move";
    }
    return;
  }
  if (gesture.type === "move") {
    const b0 = gesture.before[gesture.idx];
    const b = state.boxes[gesture.idx];
    b.x = b0.x + (ip.x - gesture.start.x);
    b.y = b0.y + (ip.y - gesture.start.y);
    // la box reste entière dans l'image pendant le déplacement
    b.x = Math.min(Math.max(0, b.x), state.current.w - b.w);
    b.y = Math.min(Math.max(0, b.y), state.current.h - b.h);
    draw();
    return;
  }
  if (gesture.type === "resize") {
    const b0 = gesture.before[state.selected];
    const dx = ip.x - gesture.start.x, dy = ip.y - gesture.start.y;
    let x0 = b0.x, y0 = b0.y, x1 = b0.x + b0.w, y1 = b0.y + b0.h;
    const h = gesture.handle;
    if (h.includes("w")) x0 += dx;
    if (h.includes("e")) x1 += dx;
    if (h.includes("n")) y0 += dy;
    if (h.includes("s")) y1 += dy;
    const nb = clampBox({ ...b0, x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) });
    state.boxes[state.selected] = nb;
    gesture.moved = true;
    draw();
  }
});

function endPointer(e) {
  if (!pointers.has(e.pointerId)) return;
  pointers.delete(e.pointerId);

  if (gesture && gesture.type === "pinch") {
    // le pinch se termine quand il reste < 2 doigts ; on ignore le doigt restant
    if (pointers.size < 2) gesture = pointers.size === 1 ? { type: "dead" } : null;
    return;
  }
  if (gesture && gesture.type === "dead") {
    if (pointers.size === 0) gesture = null;
    return;
  }
  if (!gesture) return;

  const p = pointerPos(e);

  if (gesture.type === "draw" && draftBox) {
    const r = normRect(draftBox);
    draftBox = null;
    if (r.w * scaleTotal() >= 4 || r.h * scaleTotal() >= 4) {
      pushHistory();
      state.boxes.push(clampBox({ ...r, label_id: state.currentClass }));
    }
    gesture = null;
    draw();
    return;
  }
  if (gesture.type === "maybe-draw") {
    // tap sur zone vide : désélection + détection du double-tap (recadrage)
    const now = Date.now();
    if (now - lastTap.t < 320 && Math.hypot(p.x - lastTap.x, p.y - lastTap.y) < 42) {
      fitView();
      lastTap.t = 0;
    } else {
      lastTap = { t: now, x: p.x, y: p.y };
    }
    state.selected = -1;
    gesture = null;
    draw();
    return;
  }
  if (gesture.type === "move") {
    const moved = state.boxes[gesture.idx];
    state.boxes[gesture.idx] = gesture.before[gesture.idx]; // état pré-geste
    pushHistory();
    state.boxes[gesture.idx] = clampBox(moved);
    gesture = null;
    draw();
    return;
  }
  if (gesture.type === "resize" && gesture.moved) {
    const resized = state.boxes[state.selected];
    state.boxes[state.selected] = gesture.before[state.selected];
    pushHistory();
    state.boxes[state.selected] = resized;
    gesture = null;
    draw();
    return;
  }
  gesture = null; // maybe-move sans mouvement = simple sélection (déjà faite)
}
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);

// Molette : zoom (confort en test desktop / tablette avec souris)
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const p = pointerPos(e);
  const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  const newZoom = Math.min(Math.max(view.zoom * factor, 0.5), 14);
  const s0 = scaleTotal();
  const origin = { x: (p.x - view.panX) / s0, y: (p.y - view.panY) / s0 };
  const s1 = view.baseScale * newZoom;
  view.zoom = newZoom;
  view.panX = p.x - origin.x * s1;
  view.panY = p.y - origin.y * s1;
  draw();
}, { passive: false });

/* ══════════════════ CLASSES ══════════════════ */

function renderClassChips() {
  const box = $("class-chips");
  box.innerHTML = "";
  state.config.classes.forEach((name, i) => {
    const btn = document.createElement("button");
    btn.className = "class-chip" + (i === state.currentClass ? " active" : "");
    btn.style.setProperty("--cc", state.config.colors[name] || "#888");
    btn.innerHTML = `<span class="num">${i + 1}</span>${name}`;
    btn.onclick = () => setCurrentClass(i, true);
    box.appendChild(btn);
  });
}

function setCurrentClass(id, applyToSelection) {
  state.currentClass = id;
  if (applyToSelection && state.selected >= 0) {
    pushHistory();
    state.boxes[state.selected].label_id = id;
  }
  renderClassChips();
  draw();
}

/* ══════════════════ CHARGEMENT / FILE D'IMAGES ══════════════════ */

let timerInterval = null;

function startEditorTimer() {
  state.startTime = Date.now();
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    $("editor-timer").textContent = "⏱ " + fmtDuration((Date.now() - state.startTime) / 1000);
  }, 1000);
  $("editor-timer").textContent = "⏱ 0:00";
}

async function openNextInQueue() {
  if (state.current && state.current.url) URL.revokeObjectURL(state.current.url);
  state.current = null;

  if (!state.queue.length) {
    clearInterval(timerInterval);
    const done = state.queueTotal;
    state.queueTotal = 0;
    showView("home");
    refreshStats();
    if (done > 0) toast(`✅ Série terminée — ${done} image(s) traitée(s)`);
    return;
  }

  const item = state.queue.shift();
  const token = ++state.imageToken;
  state.boxes = [];
  state.history = [];
  state.future = [];
  state.selected = -1;
  draftBox = null;
  state.highlight = false;
  $("btn-highlight").classList.remove("on");

  const url = URL.createObjectURL(item.file);
  const img = new Image();
  try {
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
  } catch {
    URL.revokeObjectURL(url);
    toast(`Image illisible : ${item.name}`, true);
    openNextInQueue();
    return;
  }
  if (token !== state.imageToken) { URL.revokeObjectURL(url); return; }

  state.current = {
    ...item,
    img,
    url,
    w: img.naturalWidth,
    h: img.naturalHeight,
  };

  $("editor-name").textContent = item.name;
  const done = state.queueTotal - state.queue.length;
  $("editor-progress").textContent = state.queueTotal > 1 ? `Image ${done}/${state.queueTotal}` : (item.fromDataset ? "Vérification" : "");
  showView("editor");
  fitView();
  draw();
  startEditorTimer();

  // Hint gestuel : visible 4 s sur la première image
  $("canvas-hint").classList.remove("hidden");
  setTimeout(() => $("canvas-hint").classList.add("hidden"), 4000);

  // Pré-remplissage : label existant (vérification) ou prédiction IA (nouvelles images)
  if (item.fromDataset) {
    try {
      const r = await fetch(`/api/label/${encodeURIComponent(item.name)}`);
      const data = await r.json();
      if (token !== state.imageToken) return;
      if (data.exists) {
        state.boxes = data.boxes.map((b) => ({ x: b.bbox[0], y: b.bbox[1], w: b.bbox[2], h: b.bbox[3], label_id: b.label_id }));
        draw();
      } else {
        toast("Aucun label existant — dessinez ou lancez l'IA 🤖");
      }
    } catch {
      toast("Impossible de charger le label existant", true);
    }
  } else {
    runPredict(false);
  }
}

async function runPredict(fromButton) {
  if (!state.current) return;
  const token = state.imageToken;
  $("spinner").hidden = false;
  $("spinner-text").textContent = "Analyse IA…";
  try {
    const fd = new FormData();
    fd.append("file", state.current.file, state.current.name);
    fd.append("conf", String(state.conf));
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    if (token !== state.imageToken) return;
    if (fromButton && state.boxes.length) pushHistory();
    state.boxes = data.boxes.map((b) => ({ x: b.bbox[0], y: b.bbox[1], w: b.bbox[2], h: b.bbox[3], label_id: b.label_id }));
    state.selected = -1;
    draw();
    if (!state.boxes.length) toast("IA : aucun objet détecté — annotez à la main");
  } catch {
    if (token === state.imageToken) toast("Pré-annotation IA indisponible", true);
  } finally {
    if (token === state.imageToken) $("spinner").hidden = true;
  }
}

/* ══════════════════ SAUVEGARDE ══════════════════ */

async function validate() {
  if (!state.current || state.saving) return;
  state.saving = true;
  $("btn-validate").disabled = true;
  $("spinner").hidden = false;
  $("spinner-text").textContent = "Sauvegarde…";
  try {
    const fd = new FormData();
    fd.append("file", state.current.file, state.current.name);
    fd.append("annotations", JSON.stringify(
      state.boxes.map((b) => ({ bbox: [b.x, b.y, b.w, b.h], label_id: b.label_id }))
    ));
    fd.append("name", state.current.name);
    fd.append("session_id", state.sessionId);
    fd.append("duration_sec", String((Date.now() - state.startTime) / 1000));
    fd.append("overwrite", state.current.fromDataset ? "true" : "false");
    const r = await fetch("/api/save", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    toast(`✅ ${data.saved_name} — ${data.nb_boxes} bbox (⏱ ${fmtDuration((Date.now() - state.startTime) / 1000)})`);
    openNextInQueue();
  } catch {
    toast("Échec de la sauvegarde — vérifiez la connexion au serveur", true);
  } finally {
    state.saving = false;
    $("btn-validate").disabled = false;
    $("spinner").hidden = true;
  }
}

/* ══════════════════ GALERIE (VÉRIFICATION) ══════════════════ */

async function openGallery() {
  showView("gallery");
  const grid = $("gallery-grid");
  grid.innerHTML = "<p class='gallery-empty'>Chargement…</p>";
  try {
    const r = await fetch("/api/gallery");
    const data = await r.json();
    renderGallery(data.images);
  } catch {
    grid.innerHTML = "<p class='gallery-empty'>Serveur injoignable</p>";
  }
}

let galleryItems = [];
function renderGallery(items) {
  if (items) galleryItems = items;
  const grid = $("gallery-grid");
  const filtered = galleryItems.filter((it) =>
    state.galleryFilter === "all" ||
    (state.galleryFilter === "annotated" && it.annotated) ||
    (state.galleryFilter === "unannotated" && !it.annotated)
  );
  grid.innerHTML = "";
  if (!filtered.length) {
    grid.innerHTML = "<p class='gallery-empty'>Aucune image dans cette catégorie</p>";
    return;
  }
  for (const it of filtered) {
    const btn = document.createElement("button");
    btn.className = "thumb";
    btn.innerHTML =
      `<img loading="lazy" src="/api/thumb/${encodeURIComponent(it.name)}" alt="">` +
      `<span class="thumb-badge">${it.annotated ? "✅" : "⬜"}</span>` +
      `<span class="thumb-name">${it.name}</span>`;
    btn.onclick = () => openFromDataset(it.name);
    grid.appendChild(btn);
  }
}

async function openFromDataset(name) {
  try {
    const r = await fetch(`/api/image/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error();
    const blob = await r.blob();
    state.queue = [{ file: new File([blob], name, { type: blob.type }), name, fromDataset: true }];
    state.queueTotal = 1;
    openNextInQueue();
  } catch {
    toast("Impossible de charger cette image", true);
  }
}

/* ══════════════════ STATS ACCUEIL ══════════════════ */

async function refreshStats() {
  try {
    const r = await fetch("/api/stats");
    const s = await r.json();
    $("stat-images").textContent = s.images;
    $("stat-annotated").textContent = s.annotated;
    $("stat-today").textContent = s.today;
    $("stat-mean").textContent = s.mean_sec_today != null ? `${Math.round(s.mean_sec_today)} s` : "–";
    $("server-status").textContent = "🟢 Serveur connecté · modèle YOLO prêt";
  } catch {
    $("server-status").textContent = "🔴 Serveur injoignable — vérifiez le WiFi et que server.py tourne sur le PC";
  }
}

/* ══════════════════ ÉVÈNEMENTS UI ══════════════════ */

$("btn-camera").onclick = () => $("input-camera").click();
$("btn-import").onclick = () => $("input-import").click();
$("btn-verify").onclick = openGallery;

function filesToQueue(fileList) {
  const files = [...fileList].filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  const ts = () => new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  state.queue = files.map((f, i) => ({
    file: f,
    // les photos caméra arrivent souvent sans nom exploitable → nom horodaté
    name: f.name && f.name !== "image.jpg" ? f.name : `mob_${ts()}_${i + 1}.jpg`,
    fromDataset: false,
  }));
  state.queueTotal = state.queue.length;
  openNextInQueue();
}

$("input-camera").onchange = (e) => { filesToQueue(e.target.files); e.target.value = ""; };
$("input-import").onchange = (e) => { filesToQueue(e.target.files); e.target.value = ""; };

$("editor-back").onclick = () => {
  const remaining = state.queue.length;
  if (remaining && !confirm(`Abandonner la série ? (${remaining} image(s) restante(s))`)) return;
  state.queue = [];
  state.queueTotal = 0;
  state.imageToken++;
  clearInterval(timerInterval);
  if (state.current && state.current.url) URL.revokeObjectURL(state.current.url);
  const wasDataset = state.current && state.current.fromDataset;
  state.current = null;
  if (wasDataset) openGallery(); else { showView("home"); refreshStats(); }
};

$("gallery-back").onclick = () => { showView("home"); refreshStats(); };

for (const chip of document.querySelectorAll(".filter-chips .chip")) {
  chip.onclick = () => {
    document.querySelectorAll(".filter-chips .chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    state.galleryFilter = chip.dataset.filter;
    renderGallery();
  };
}

$("btn-undo").onclick = undo;
$("btn-redo").onclick = redo;
$("btn-delete").onclick = deleteSelected;
$("btn-highlight").onclick = () => {
  state.highlight = !state.highlight;
  $("btn-highlight").classList.toggle("on", state.highlight);
  draw();
};
$("btn-predict").onclick = () => runPredict(true);
$("btn-validate").onclick = validate;

$("conf-slider").oninput = (e) => {
  state.conf = parseFloat(e.target.value);
  $("conf-value").textContent = state.conf.toFixed(2);
};

// Raccourcis clavier — parité avec la version PC (tablette + clavier, tests desktop)
window.addEventListener("keydown", (e) => {
  if (!document.querySelector("#view-editor.active")) return;
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if (e.key === "Enter") validate();
  if (e.key === "Delete" || e.key === "Backspace") deleteSelected();
  if (e.key === "Escape") { state.selected = -1; draw(); }
  if (e.key === "h" || e.key === "H") $("btn-highlight").click();
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "y" || (e.shiftKey && e.key.toLowerCase() === "z"))) { e.preventDefault(); redo(); }
  const num = parseInt(e.key);
  if (!isNaN(num) && num >= 1 && num <= (state.config ? state.config.classes.length : 0)) {
    setCurrentClass(num - 1, true);
  }
});

/* ══════════════════ INITIALISATION ══════════════════ */

async function init() {
  refreshStats();
  try {
    const r = await fetch("/api/config");
    state.config = await r.json();
    state.conf = state.config.conf_default;
    $("conf-slider").value = state.conf;
    $("conf-value").textContent = state.conf.toFixed(2);
    renderClassChips();
  } catch {
    toast("Impossible de joindre le serveur", true);
  }

  // Service worker (PWA) — actif seulement en contexte sécurisé
  // (HTTPS ou localhost) ; en HTTP sur le LAN l'app reste 100% fonctionnelle.
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }
}

init();
