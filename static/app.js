const state = {
  page: "dashboard",
  chart: null,
};

const labelMap = {
  pass: "合格",
  fail: "不合格",
  review: "待复核",
  no_signal: "无信号",
  pending: "待处理",
};

const stateMap = {
  idle: "空闲",
  running: "巡检中",
  finished: "已完成",
  error: "出错",
};

const stepMap = {
  "抓图": "正在抓图",
  "下载": "正在下载",
  "分析": "正在分析",
  "入库": "写入结果",
  "生成演示图": "生成演示图",
};

const EMPTY_IMG = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || resp.statusText);
  }
  return resp.json();
}

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 2400);
}

function pill(decision) {
  return `<span class="pill ${decision}">${labelMap[decision] || decision}</span>`;
}

function fmtTime(t) {
  if (!t) return "-";
  return String(t).replace("T", " ").slice(0, 19);
}

function fmtDuration(seconds) {
  if (!seconds || seconds <= 0) return "-";
  const s = Math.round(seconds);
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  return `${Math.floor(s / 3600)} 小时 ${Math.floor((s % 3600) / 60)} 分`;
}

function refreshStartButton(p) {
  const btn = document.getElementById("startCycle");
  const running = Boolean(p && p.state === "running");
  btn.disabled = running;
  btn.innerHTML = running ? "◉ 巡检中…" : "▶ 开始一轮巡检";
  const sys = document.getElementById("sysState");
  if (sys) sys.textContent = running ? "任务运行中" : "系统在线";
}

function switchPage(page) {
  state.page = page;
  document.querySelectorAll("nav button").forEach((b) => {
    b.classList.toggle("active", b.dataset.page === page);
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("active", p.id === `page-${page}`);
  });
  loadPage(page);
}

async function loadPage(page) {
  if (page === "dashboard") await loadDashboard();
  if (page === "cameras") await loadCameras();
  if (page === "captures") await loadCaptures();
  if (page === "review") await loadReview();
  if (page === "settings") await loadSettings();
}

async function loadDashboard() {
  const [stats, captures, progress] = await Promise.all([
    api("/api/stats"),
    api("/api/captures?limit=12"),
    api("/api/cycle/progress"),
  ]);
  const by = stats.by_decision || {};
  document.getElementById("statCameras").textContent = stats.total_cameras;
  document.getElementById("statCaptures").textContent = stats.total_captures;
  document.getElementById("statPass").textContent = by.pass || 0;
  document.getElementById("statFail").textContent = by.fail || 0;
  document.getElementById("statReview").textContent = by.review || 0;
  document.getElementById("statNoSignal").textContent = by.no_signal || 0;
  document.getElementById("reviewBadge").textContent = by.review || 0;

  renderChart(by);
  const cycle = stats.cycle || {};
  const p = progress || {};
  const percent = Number(p.percent || 0);
  document.getElementById("cycleState").textContent = stateMap[p.state || cycle.state] || "空闲";
  document.getElementById("cycleStateTag").textContent = (p.state || cycle.state || "idle").toUpperCase();
  document.getElementById("cyclePercent").textContent = `${percent.toFixed(1)}%`;
  document.getElementById("cycleProgress").textContent = `${p.processed || 0} / ${p.total || 0}（${percent.toFixed(1)}%）`;
  document.getElementById("cycleBar").style.width = `${Math.max(0, Math.min(100, percent))}%`;
  document.getElementById("cycleRate").textContent = p.rate_per_sec ? `${p.rate_per_sec} 路/秒` : "-";
  document.getElementById("cycleEta").textContent = p.state === "running" ? fmtDuration(p.eta_seconds) : "-";
  document.getElementById("cycleStarted").textContent = fmtTime(cycle.started_at);
  document.getElementById("cycleFinished").textContent = fmtTime(cycle.finished_at);
  document.getElementById("headCycleState").textContent = stateMap[p.state || cycle.state] || "空闲";
  document.getElementById("headReview").textContent = by.review || 0;
  renderActiveTasks(p.active || []);
  refreshStartButton(p);

  const strip = document.getElementById("recentList");
  strip.innerHTML = (captures.rows || []).map((c) => `
    <div class="mini">
      <img src="${c.thumb_url}" onerror="this.src=''" data-open="${c.id}" alt="抓图" />
      <span class="tag">${labelMap[c.decision] || c.decision}</span>
    </div>`).join("");
  strip.querySelectorAll("img[data-open]").forEach((img) => {
    img.addEventListener("click", () => openDetail(Number(img.dataset.open)));
  });
}

function renderActiveTasks(active) {
  const box = document.getElementById("activeTasks");
  if (!active || active.length === 0) {
    box.innerHTML = "<p class=\"muted\">暂无进行中的任务</p>";
    return;
  }
  box.innerHTML = active.map((t) => `
    <div class="active-task">
      <div class="active-task-name">
        <span class="active-task-step">${stepMap[t.step] || t.step || "处理中"}</span>
        ${t.channel_name || `${t.device_serial}-${t.channel_no}`}
      </div>
      <div class="active-task-sub">${t.device_serial} · 通道 ${t.channel_no}</div>
    </div>`).join("");
}

function renderChart(by) {
  const data = ["pass", "fail", "review", "no_signal"].map((k) => by[k] || 0);
  if (state.chart) state.chart.destroy();
  const canvas = document.getElementById("decisionChart");
  const wrap = canvas.parentElement;
  if (window.Chart) {
    canvas.style.display = "";
    const fallback = wrap.querySelector(".donut-fallback");
    if (fallback) fallback.remove();
    state.chart = new Chart(canvas, {
      type: "doughnut",
      data: {
        labels: ["合格", "不合格", "待复核", "无信号"],
        datasets: [{ data, backgroundColor: ["#16a34a", "#dc2626", "#d97706", "#9ca3af"] }],
      },
      options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } },
    });
  } else {
    canvas.style.display = "none";
    const total = data.reduce((a, b) => a + b, 0) || 1;
    const colors = ["#2dd4a7", "#ff5c7a", "#fbbf24", "#8295b7"];
    let acc = 0;
    const stops = data.map((v, i) => {
      const start = acc / total * 360;
      acc += v;
      const end = acc / total * 360;
      return `${colors[i]} ${start}deg ${end}deg`;
    }).join(", ");
    const old = wrap.querySelector(".donut-fallback");
    if (old) old.remove();
    const el = document.createElement("div");
    el.className = "donut-fallback";
    el.style.background = `conic-gradient(${stops || "#1f2937 0deg 360deg"})`;
    wrap.appendChild(el);
  }
}

async function loadCameras() {
  const rows = await api("/api/cameras?limit=500");
  document.getElementById("cameraCountLabel").textContent = rows.length;
  document.getElementById("cameraGrid").innerHTML = rows.map((c) => {
    const last = c.last_capture;
    const stateClass = last ? last.decision : "pending";
    const stateLabel = labelMap[stateClass] || "待处理";
    return `
    <div class="camera-card">
      <div class="camera-state">
        <span class="pill ${stateClass}">${stateLabel}</span>
        <span class="panel-tag">${c.scene_template || "default"}</span>
      </div>
      <h3>${c.channel_name || c.device_serial}</h3>
      <p>${c.device_serial} · 通道 ${c.channel_no}</p>
      <p>${(c.meta && c.meta.store_name) || "-"}</p>
      ${last ? `<p>最近置信度 ${(last.confidence || 0).toFixed(2)}</p>` : ""}
    </div>`;
  }).join("") || "<p class=\"muted\">暂无相机</p>";
}

async function loadCaptures() {
  const filter = document.getElementById("captureFilter").value;
  const query = filter ? `&decision=${filter}` : "";
  const data = await api(`/api/captures?limit=200${query}`);
  const rows = data.rows || [];
  const countBy = rows.reduce((acc, c) => {
    acc[c.decision] = (acc[c.decision] || 0) + 1;
    return acc;
  }, {});
  document.getElementById("captureSummary").textContent =
    `当前显示 ${rows.length} 条 · 合格 ${countBy.pass || 0} · 不合格 ${countBy.fail || 0} · 待复核 ${countBy.review || 0} · 无信号 ${countBy.no_signal || 0}`;
  document.getElementById("captureGrid").innerHTML = rows.map((c) => {
    const cam = c.camera || {};
    return `
    <div class="capture-card">
      <div class="capture-thumb">
        <img src="${c.thumb_url}" alt="抓图" />
        <span class="tag">${labelMap[c.decision] || c.decision}</span>
      </div>
      <div class="capture-meta">
        <h3>${cam.channel_name || cam.device_serial || c.camera_id}</h3>
        <p>${cam.device_serial || "-"} · 通道 ${cam.channel_no || "-"}</p>
        <p class="capture-time">${fmtTime(c.capture_time)}</p>
        <p class="capture-reason">${(c.reasons || []).join("；") || "无"}</p>
        <div class="capture-actions">
          <span>置信度 ${(c.confidence || 0).toFixed(2)}</span>
          <button class="btn" data-open="${c.id}">查看</button>
        </div>
      </div>
    </div>`;
  }).join("") || "<p class=\"muted\">暂无抓图记录</p>";
  document.querySelectorAll("button[data-open]").forEach((b) => {
    b.addEventListener("click", () => openDetail(Number(b.dataset.open)));
  });
}

async function loadReview() {
  const data = await api("/api/captures?decision=review&limit=200");
  document.getElementById("reviewList").innerHTML = (data.rows || []).map((c) => `
    <div class="review-item">
      <figure class="review-figure">
        <figcaption>当前抓图</figcaption>
        <img src="${c.image_url}" alt="当前抓图" />
      </figure>
      <figure class="review-figure">
        <figcaption>基准图</figcaption>
        <img src="${c.baseline_url || EMPTY_IMG}" alt="基准图" />
      </figure>
      <div class="meta">
        <h3>${c.camera ? c.camera.channel_name : c.camera_id}</h3>
        <p>序列号：${c.camera ? c.camera.device_serial : "-"} 通道 ${c.camera ? c.camera.channel_no : "-"}</p>
        <p>门店：${c.camera && c.camera.meta ? c.camera.meta.store_name || "-" : "-"}</p>
        <p>时间：${fmtTime(c.capture_time)}</p>
        <p>置信度：${(c.confidence || 0).toFixed(2)}</p>
        <p>原因：${(c.reasons || []).join("；")}</p>
        <div class="actions">
          <button class="btn good" data-review="${c.id}" data-decision="pass">合格</button>
          <button class="btn bad" data-review="${c.id}" data-decision="fail">不合格</button>
          <button class="btn warn" data-review="${c.id}" data-decision="pass" data-baseline="1">合格并设为基准</button>
        </div>
      </div>
    </div>`).join("") || "<p>暂无待复核记录</p>";
  document.querySelectorAll("button[data-review]").forEach((b) => {
    b.addEventListener("click", () => {
      submitReview(
        Number(b.dataset.review),
        b.dataset.decision,
        b.dataset.baseline === "1",
      );
    });
  });
}

async function submitReview(id, decision, setBaseline) {
  try {
    await api(`/api/captures/${id}/review`, {
      method: "POST",
      body: JSON.stringify({ decision, set_baseline: setBaseline, note: "" }),
    });
    toast("复核已提交");
    loadReview();
    loadDashboard();
  } catch (err) {
    toast(`操作失败：${err.message}`);
  }
}

async function openDetail(id) {
  const detail = await api(`/api/captures/${id}`);
  document.getElementById("detailImage").src = detail.image_url;
  document.getElementById("detailBaseline").src = detail.baseline_url || EMPTY_IMG;
  document.getElementById("detailInfo").innerHTML = `
    <h3>判定详情</h3>
    <p>判定：${pill(detail.decision)} 置信度：${(detail.confidence || 0).toFixed(2)}</p>
    <p>原因：${(detail.reasons || []).join("；")}</p>
    <p>指标：${Object.entries(detail.metrics || {}).map(([k, v]) => `${k}=${v}`).join("， ")}</p>
    <div class="actions">
      <button class="btn good" data-review="${detail.id}" data-decision="pass">复核合格</button>
      <button class="btn bad" data-review="${detail.id}" data-decision="fail">复核不合格</button>
      <button class="btn warn" data-review="${detail.id}" data-decision="pass" data-baseline="1">合格并设为基准</button>
    </div>`;
  document.getElementById("detailModal").classList.remove("hidden");
  document.querySelectorAll("#detailInfo button[data-review]").forEach((b) => {
    b.addEventListener("click", async () => {
      await submitReview(Number(b.dataset.review), b.dataset.decision, b.dataset.baseline === "1");
      document.getElementById("detailModal").classList.add("hidden");
    });
  });
}

async function loadSettings() {
  const s = await api("/api/settings");
  document.getElementById("settingDemo").value = s.demo_mode ? "是" : "否";
  document.getElementById("settingConcurrency").value = s.concurrency;
  document.getElementById("settingInterval").value = s.capture_interval;
  document.getElementById("settingConfidence").value = s.thresholds.confidence_review;
  document.getElementById("settingRollFail").value = s.thresholds.roll_fail_deg;
  document.getElementById("settingBaseRot").value = s.thresholds.baseline_rotation_deg;
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("nav button").forEach((b) => {
    b.addEventListener("click", () => switchPage(b.dataset.page));
  });
  document.querySelectorAll("[data-close]").forEach((el) => {
    el.addEventListener("click", () => {
      document.getElementById(el.dataset.close).classList.add("hidden");
    });
  });
  document.getElementById("captureFilter").addEventListener("change", loadCaptures);
  document.getElementById("startCycle").addEventListener("click", async () => {
    const btn = document.getElementById("startCycle");
    btn.disabled = true;
    btn.innerHTML = "◉ 启动中…";
    try {
      await api("/api/cycle/start", { method: "POST" });
      toast("巡检已启动");
      btn.innerHTML = "◉ 巡检中…";
      setTimeout(loadDashboard, 1000);
    } catch (err) {
      toast(`启动失败：${err.message}`);
      btn.disabled = false;
      btn.innerHTML = "▶ 开始一轮巡检";
    }
  });
  document.getElementById("saveSettings").addEventListener("click", async () => {
    try {
      await api("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          thresholds: {
            confidence_review: Number(document.getElementById("settingConfidence").value),
            roll_fail_deg: Number(document.getElementById("settingRollFail").value),
            baseline_rotation_deg: Number(document.getElementById("settingBaseRot").value),
          },
        }),
      });
      toast("设置已保存");
    } catch (err) {
      toast(`保存失败：${err.message}`);
    }
  });
  document.getElementById("seedDemo").addEventListener("click", async () => {
    if (!confirm("重置演示数据会清空演示环境，确认继续？")) return;
    try {
      await api("/api/demo/seed", { method: "POST" });
      toast("演示数据已刷新");
      loadDashboard();
    } catch (err) {
      toast(`操作失败：${err.message}`);
    }
  });

  switchPage("dashboard");
  setInterval(() => {
    if (state.page === "dashboard") loadDashboard();
  }, 2000);
  setInterval(async () => {
    try {
      const p = await api("/api/cycle/progress");
      refreshStartButton(p);
    } catch (err) {
      // 忽略瞬时错误，下一次轮询会再试。
    }
  }, 3000);
});
