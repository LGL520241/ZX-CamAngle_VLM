function storedView(kind) {
  try {
    const value = localStorage.getItem(`camangle.view.${kind}`);
    return ["list", "dense"].includes(value) ? value : "grid";
  } catch (err) {
    return "grid";
  }
}

const state = {
  page: "dashboard",
  chart: null,
  batches: [],
  batchId: null,
  batchAll: false,
  cameraPage: 1,
  cameraTotal: 0,
  capturePage: 1,
  captureTotal: 0,
  reviewPage: 1,
  reviewTotal: 0,
  detailIds: [],
  detailIndex: -1,
  detailRequestId: 0,
  storeRows: [],
  storeFilter: "",
  storeName: "",
  searchTimers: {},
  settingsSnapshot: null,
  views: {
    cameras: storedView("cameras"),
    captures: storedView("captures"),
    review: storedView("review"),
  },
};

const CAMERA_PAGE_SIZE = 24;
const CAMERA_LIST_PAGE_SIZE = 40;
const CAMERA_DENSE_PAGE_SIZE = 60;
const CAPTURE_PAGE_SIZE = 16;
const CAPTURE_LIST_PAGE_SIZE = 24;
const CAPTURE_DENSE_PAGE_SIZE = 40;
const REVIEW_PAGE_SIZE = 8;
const REVIEW_LIST_PAGE_SIZE = 16;
const REVIEW_DENSE_PAGE_SIZE = 24;
const THRESHOLD_PRESETS = {
  occlusion: {
    settingOcclusionArea: "35",
    settingOcclusionDark: "32",
    settingOcclusionGray: "35",
    settingOcclusionTexture: "10",
  },
  blur: {
    settingBlurScore: "60",
    settingBlurRatio: "50",
  },
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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function fmtPercent(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
}

function renderOcclusion(result) {
  if (!result) return '<p class="occlusion-note">此历史记录没有遮挡检测结果，下一轮巡检会生成新结果。</p>';
  if (result.status === "skipped") {
    return `<section class="occlusion-result"><div class="occlusion-result-head"><h3>遮挡面积候选</h3><span class="pill pending">已关闭</span></div><p class="occlusion-note">${escapeHtml(result.reason || "该模块未启用")}</p></section>`;
  }
  const metrics = result.metrics || {};
  const status = { pass: "pass", fail: "fail", warn: "review", unknown: "review" }[result.status] || "review";
  const regions = Array.isArray(result.regions) ? result.regions : [];
  return `
    <section class="occlusion-result">
      <div class="occlusion-result-head"><h3>遮挡检测</h3>${pill(status)}</div>
      <p>${escapeHtml(result.reason || "无检测说明")}</p>
      <div class="occlusion-metrics">
        <div><span>黑灰低纹理候选面积</span><strong>${fmtPercent(metrics.candidate_ratio)}</strong></div>
        <div><span>黑灰低纹理区域占比</span><strong>${fmtPercent(metrics.occlusion_ratio)}</strong></div>
        <div><span>最大连续区域</span><strong>${fmtPercent(metrics.largest_region_ratio)}</strong></div>
        <div><span>面积阈值</span><strong>${fmtPercent(metrics.threshold)}</strong></div>
      </div>
      <p>置信度：${Number.isFinite(result.confidence) ? result.confidence.toFixed(2) : "-"}</p>
      <p>其中黑色区域：${fmtPercent(metrics.black_region_ratio)} · 灰色区域：${fmtPercent(metrics.gray_region_ratio)}${Number(metrics.letterbox_ratio) > 0.001 ? ` · 已忽略画面黑边：${fmtPercent(metrics.letterbox_ratio)}` : ""}</p>
      ${regions.length ? `<p>检测区域：${regions.length} 处${regions.slice(0, 5).map((region, index) => ` · 区域 ${index + 1} 占 ${fmtPercent(region.area_ratio)}`).join("")}${regions.length > 5 ? " …" : ""}</p>` : ""}
      <p class="occlusion-note">按当前抓图的黑灰低纹理区域面积判定。正常灰墙、地面、设备表面也可能触发，请结合原图复核。</p>
    </section>`;
}

function renderBlur(result) {
  if (!result) return '<p class="occlusion-note">此历史记录没有分区清晰度结果，下一轮巡检会生成新结果。</p>';
  if (result.status === "skipped") {
    return `<section class="occlusion-result"><div class="occlusion-result-head"><h3>模糊程度</h3><span class="pill pending">已关闭</span></div><p class="occlusion-note">${escapeHtml(result.reason || "该模块未启用")}</p></section>`;
  }
  const metrics = result.metrics || {};
  const status = { pass: "pass", fail: "fail", warn: "review", unknown: "review" }[result.status] || "review";
  const score = Number.isFinite(metrics.blur_score) ? metrics.blur_score.toFixed(1) : "-";
  return `<section class="occlusion-result">
    <div class="occlusion-result-head"><h3>模糊检测</h3>${pill(status)}</div>
    <p>${escapeHtml(result.reason || "无检测说明")}</p>
    <div class="occlusion-metrics">
      <div><span>主体清晰度分数</span><strong>${score}</strong></div>
      <div><span>低清晰度分区占比</span><strong>${fmtPercent(metrics.blur_affected_ratio)}</strong></div>
      <div><span>有效分区数量</span><strong>${escapeHtml(metrics.blur_informative_tiles ?? "-")}</strong></div>
      <div><span>有效检查面积</span><strong>${fmtPercent(metrics.blur_coverage)}</strong></div>
    </div>
    <p class="occlusion-note">清晰度分数越高通常越清楚；已排除黑灰异常区域。纯色画面、强噪声或结构不足会转人工确认。</p>
  </section>`;
}

function renderSemantic(result, elementResult) {
  if (!result) return '<p class="occlusion-note">此历史记录没有 Qwen 语义巡检结果。</p>';
  if (result.status === "skipped") {
    return `<section class="occlusion-result"><div class="occlusion-result-head"><h3>Qwen 场景与元素命中</h3><span class="pill pending">已关闭</span></div><p class="occlusion-note">${escapeHtml(result.reason || "Qwen 语义模块未启用")}</p></section>`;
  }
  const status = { pass: "pass", fail: "fail", warn: "review", unknown: "review" }[result.status] || "review";
  const elements = Array.isArray(result.visible_elements) ? result.visible_elements : [];
  const hits = elements.filter((item) => item.visible && Number(item.confidence || 0) >= 0.55);
  const score = Number(elementResult && elementResult.score);
  const elementApplicable = !elementResult || elementResult.status !== "skipped";
  const subjectAffected = Boolean(result.occlusion_affects_subject || result.monitoring_subject_affected);
  const viewLabels = {
    store_outdoor: "店外", store_entrance: "门口人流", kitchen: "后厨",
    cashier: "收银", kitchen_cashier: "后厨+收银",
    dining: "用餐区", storage: "仓库", other_indoor: "其他室内", unknown: "未知",
  };
  const outdoorLabels = { proper: "正对门店/通道", facing_sky: "对天", facing_wall: "对墙", unknown: "未知" };
  return `<section class="occlusion-result">
    <div class="occlusion-result-head"><h3>Qwen 场景与元素命中</h3>${pill(status)}</div>
    <p>${escapeHtml(result.reason || "无场景说明")}</p>
    <div class="occlusion-metrics">
      <div><span>场景类型</span><strong>${escapeHtml(viewLabels[result.camera_type] || result.camera_type || "-")}</strong></div>
      <div><span>元素命中率</span><strong>${elementApplicable && Number.isFinite(score) ? fmtPercent(score) : "不适用"}</strong></div>
      <div><span>遮挡判断</span><strong>${escapeHtml({ none: "未发现", partial: "局部", severe: "严重", unknown: "不确定" }[result.occlusion] || "-")}</strong></div>
      <div><span>Qwen 遮挡面积</span><strong>${fmtPercent(Number(result.occlusion_area_ratio))}</strong></div>
      <div><span>主体是否受影响</span><strong>${subjectAffected ? "是" : "否"}</strong></div>
      <div><span>模糊判断</span><strong>${escapeHtml({ none: "清晰", slight: "轻微", severe: "严重", unknown: "不确定" }[result.blur] || "-")}</strong></div>
      <div><span>人员操作</span><strong>${escapeHtml({ clear: "可判断", partial: "部分受影响", obscured: "无法判断", not_applicable: "不适用", unknown: "不确定" }[result.operation_visibility] || "-")}</strong></div>
      <div><span>店外朝向</span><strong>${escapeHtml(outdoorLabels[result.outdoor_view] || result.outdoor_view || "-")}</strong></div>
    </div>
    <p>命中元素：${hits.length ? hits.map((item) => `${escapeHtml(item.label)}（${Number(item.confidence).toFixed(2)}）`).join("、") : "无"}</p>
    ${result.angle_evidence ? `<p>角度证据：${escapeHtml(result.angle_evidence)}</p>` : ""}
    ${result.occlusion_evidence ? `<p>遮挡证据：${escapeHtml(result.occlusion_evidence)}</p>` : ""}
    <p class="occlusion-note">元素越多且置信度越高，业务区域判断越可靠；低命中率会转人工，不会直接判为合格。</p>
  </section>`;
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    cache: "no-store",
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

function fmtRelativeTime(t) {
  if (!t) return "-";
  const time = new Date(t).getTime();
  if (!Number.isFinite(time)) return fmtTime(t);
  const seconds = Math.max(0, Math.floor((Date.now() - time) / 1000));
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  return fmtTime(t);
}

function fmtDuration(seconds) {
  if (!seconds || seconds <= 0) return "-";
  const s = Math.round(seconds);
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  return `${Math.floor(s / 3600)} 小时 ${Math.floor((s % 3600) / 60)} 分`;
}

function matchesQuery(values, query) {
  const needle = String(query || "").trim().toLowerCase();
  if (!needle) return true;
  return values.filter(Boolean).join(" ").toLowerCase().includes(needle);
}

function loadingCards(count) {
  return Array.from({ length: count }, () => `
    <div class="loading-card" aria-hidden="true">
      <i></i><span></span><span></span>
    </div>`).join("");
}

function recentRowLimit() {
  const grid = document.getElementById("recentList");
  if (!grid) return 8;
  if (window.matchMedia("(max-width: 600px)").matches) return 2;
  const panel = grid.closest(".recent-panel");
  const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  const fallbackWidth = viewportWidth > 860
    ? viewportWidth - 228 - 60 - 36
    : viewportWidth - 28 - 36;
  const width = grid.clientWidth
    || (panel ? panel.clientWidth - 36 : 0)
    || fallbackWidth;
  const gap = Number.parseFloat(getComputedStyle(grid).columnGap) || 12;
  const minimum = 210;
  const columns = Math.max(1, Math.floor((width + gap) / (minimum + gap)));
  return columns * 2;
}

function withBatch(params = new URLSearchParams()) {
  if (!state.batchAll && state.batchId !== null) params.set("batch_id", String(state.batchId));
  return params;
}

function currentBatchSelection() {
  return state.batchAll ? "all" : state.batchId;
}

function batchMode(batch) {
  const latest = state.batches[0];
  return latest && Number(batch.id) === Number(latest.id) ? "latest" : "history";
}

function batchModeLabel(batch) {
  return batchMode(batch) === "latest" ? "当前批次" : "历史批次";
}

function closeBatchMenu() {
  const menu = document.getElementById("batchSelectMenu");
  const trigger = document.getElementById("batchSelectButton");
  const viewAll = document.getElementById("batchViewAll");
  if (menu) menu.hidden = true;
  if (trigger) trigger.setAttribute("aria-expanded", "false");
  if (viewAll) viewAll.setAttribute("aria-expanded", "false");
}

function setBatchMenuOpen(open) {
  const menu = document.getElementById("batchSelectMenu");
  const trigger = document.getElementById("batchSelectButton");
  const viewAll = document.getElementById("batchViewAll");
  const hint = document.getElementById("batchSelectHint");
  if (!menu || !trigger || (trigger.disabled && open)) return;
  menu.hidden = !open;
  trigger.setAttribute("aria-expanded", String(open));
  if (viewAll) viewAll.setAttribute("aria-expanded", String(open));
  if (hint) hint.textContent = open ? "收起时间段" : `查看全部 ${state.batches.length} 个时间段`;
  if (open) {
    requestAnimationFrame(() => {
      menu.querySelector('.batch-option[aria-selected="true"]')?.focus();
    });
  }
}

async function chooseBatch(batchId) {
  const selectAll = batchId === "all";
  if (selectAll && state.batchAll) {
    closeBatchMenu();
    return;
  }
  if (!selectAll && !state.batchAll && Number(batchId) === state.batchId) {
    closeBatchMenu();
    return;
  }
  state.batchAll = selectAll;
  if (!selectAll) state.batchId = Number(batchId);
  closeBatchMenu();
  resetPaging();
  await loadBatches(selectAll ? "all" : state.batchId);
  await loadPage(state.page);
}

async function loadBatches(preferredBatchId = undefined) {
  const data = await api("/api/batches?limit=100");
  state.batches = data.rows || [];
  if (preferredBatchId === "all") {
    state.batchAll = true;
  } else if (preferredBatchId !== undefined) {
    state.batchAll = false;
  }
  let nextId = state.batchId;
  if (!state.batchAll) {
    if (preferredBatchId !== undefined) nextId = preferredBatchId;
    if (nextId !== null && !state.batches.some((batch) => Number(batch.id) === Number(nextId))) {
      nextId = null;
    }
    if (nextId === null && state.batches.length) nextId = Number(state.batches[0].id);
  }
  state.batchId = nextId === null ? null : Number(nextId);

  const selected = state.batchAll
    ? null
    : state.batches.find((batch) => Number(batch.id) === state.batchId);
  const trigger = document.getElementById("batchSelectButton");
  const selectedLabel = document.getElementById("batchSelectedLabel");
  const scopeBadge = document.getElementById("batchSelectState");
  const selectHint = document.getElementById("batchSelectHint");
  const viewAll = document.getElementById("batchViewAll");
  const viewAllCount = document.getElementById("batchViewAllCount");
  trigger.disabled = state.batches.length === 0;
  viewAll.disabled = state.batches.length === 0;
  viewAllCount.textContent = state.batches.length;
  if (document.getElementById("batchSelectMenu").hidden) {
    selectHint.textContent = `查看全部 ${state.batches.length} 个时间段`;
  }
  scopeBadge.classList.toggle("all", state.batchAll);
  if (state.batchAll) {
    selectedLabel.textContent = `全部巡检记录 · ${state.batches.length} 个时间段`;
    scopeBadge.textContent = "全部数据";
    scopeBadge.classList.add("history");
  } else if (selected) {
    const stateText = stateMap[selected.state] || selected.state || "未知";
    selectedLabel.textContent = `#${selected.id} · ${fmtTime(selected.started_at)} · ${stateText}`;
    scopeBadge.textContent = batchModeLabel(selected);
    scopeBadge.classList.toggle("history", batchMode(selected) === "history");
  } else {
    selectedLabel.textContent = "暂无抓帧批次";
    scopeBadge.textContent = "无数据";
    scopeBadge.classList.add("history");
    closeBatchMenu();
  }

  const totals = state.batches.reduce((result, batch) => {
    result.capture += Number(batch.capture_count || 0);
    result.pass += Number(batch.pass_count || 0);
    result.fail += Number(batch.fail_count || 0);
    result.review += Number(batch.review_count || 0);
    result.noSignal += Number(batch.no_signal_count || 0);
    return result;
  }, { capture: 0, pass: 0, fail: 0, review: 0, noSignal: 0 });
  document.getElementById("batchMenuCount").textContent = `全部 ${state.batches.length} 个时间段`;
  document.getElementById("batchSelectOptions").innerHTML = state.batches.length
    ? `
      <button
        class="batch-option batch-option-all"
        type="button"
        role="option"
        data-batch-scope="all"
        aria-selected="${state.batchAll}"
        aria-label="全部巡检记录，累计已抓 ${totals.capture} 条"
      >
        <span class="batch-option-main">
          <span class="batch-option-title">全部巡检记录</span>
          <i class="batch-option-tag all">跨 ${state.batches.length} 个时间段</i>
        </span>
        <span class="batch-option-stats">
          <span>累计抓取 ${totals.capture} 条</span>
          <em>合格 ${totals.pass} · 不合格 ${totals.fail} · 复核 ${totals.review} · 无信号 ${totals.noSignal}</em>
        </span>
      </button>` + state.batches.map((batch) => {
      const stateText = stateMap[batch.state] || batch.state || "未知";
      const mode = batchMode(batch);
      const selectedOption = !state.batchAll && Number(batch.id) === state.batchId;
      const title = `#${batch.id} · ${fmtTime(batch.started_at)}`;
      return `
        <button
          class="batch-option"
          type="button"
          role="option"
          data-batch-id="${batch.id}"
          data-batch-scope="batch"
          aria-selected="${selectedOption}"
          aria-label="${escapeHtml(`${title}，${stateText}，${batchModeLabel(batch)}，已抓 ${batch.capture_count || 0} 条`)}"
        >
          <span class="batch-option-main">
            <span class="batch-option-title">${escapeHtml(title)}</span>
            <i class="batch-option-tag ${mode === "latest" ? "latest" : "history"}">${mode === "latest" ? "最新" : "历史"}</i>
            <i class="batch-option-tag ${escapeHtml(batch.state || "idle")}">${escapeHtml(stateText)}</i>
          </span>
          <span class="batch-option-stats">
            <span>已抓 ${batch.capture_count || 0} 条</span>
            <em>合格 ${batch.pass_count || 0} · 不合格 ${batch.fail_count || 0} · 复核 ${batch.review_count || 0} · 无信号 ${batch.no_signal_count || 0}</em>
          </span>
        </button>`;
    }).join("")
    : '<div class="recent-empty"><div><b>暂无历史批次</b><p>开始一轮巡检后，这里会按时间段保留记录。</p></div></div>';

  const summary = document.getElementById("batchSummary");
  if (!state.batches.length) {
    summary.textContent = "暂无批次，开始巡检后会创建新批次";
  } else if (state.batchAll) {
    summary.textContent =
      `全部数据 · ${state.batches.length} 个时间段 · 累计 ${totals.capture} 条 · ` +
      `合格 ${totals.pass} / 不合格 ${totals.fail} / 待复核 ${totals.review}`;
  } else {
    summary.textContent =
      `${batchModeLabel(selected)} · ${stateMap[selected.state] || selected.state} · 已抓 ${selected.capture_count || 0} 条 · ` +
      `合格 ${selected.pass_count || 0} / 不合格 ${selected.fail_count || 0} / 待复核 ${selected.review_count || 0}`;
  }
}

function renderPagination(containerId, { total, current, pageSize, unit = "条", onPage }) {
  const container = document.getElementById(containerId);
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const page = Math.max(1, Math.min(current, pages));
  if (total === 0) {
    container.innerHTML = '<span class="pagination-info">暂无数据</span>';
    return;
  }
  if (pages === 1) {
    container.innerHTML = `<span class="pagination-info">共 ${total} ${unit} · 当前仅一页</span>`;
    return;
  }

  const candidates = new Set([1, pages, page - 1, page, page + 1]);
  const visiblePages = [...candidates].filter((value) => value >= 1 && value <= pages).sort((a, b) => a - b);
  const items = [];
  visiblePages.forEach((value, index) => {
    if (index > 0 && value - visiblePages[index - 1] > 1) items.push("ellipsis");
    items.push(value);
  });
  container.innerHTML = `
    <button class="pagination-arrow" type="button" data-page="${page - 1}" ${page === 1 ? "disabled" : ""} aria-label="上一页">‹</button>
    <span class="pagination-info">第 ${page} / ${pages} 页 · 共 ${total} ${unit}</span>
    ${items.map((item) => item === "ellipsis"
      ? '<span class="pagination-ellipsis">…</span>'
      : `<button class="pagination-page ${item === page ? "active" : ""}" type="button" data-page="${item}" ${item === page ? 'aria-current="page"' : ""}>${item}</button>`).join("")}
    <button class="pagination-arrow" type="button" data-page="${page + 1}" ${page === pages ? "disabled" : ""} aria-label="下一页">›</button>`;
  container.querySelectorAll("button[data-page]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = Number(button.dataset.page);
      if (next < 1 || next > pages || next === page) return;
      scrollToPageTop();
      onPage(next);
    });
  });
}

function resetPaging() {
  state.cameraPage = 1;
  state.capturePage = 1;
  state.reviewPage = 1;
}

function gridColumnCount(containerId, minCardWidth) {
  const grid = document.getElementById(containerId);
  if (!grid) return 1;
  const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  const fallbackWidth = viewportWidth > 860 ? viewportWidth - 288 : viewportWidth - 28;
  const width = grid.clientWidth || fallbackWidth;
  const gap = Number.parseFloat(getComputedStyle(grid).columnGap) || 12;
  return Math.max(1, Math.floor((width + gap) / (minCardWidth + gap)));
}

function fullRowPageSize(containerId, minCardWidth, targetCards) {
  const columns = gridColumnCount(containerId, minCardWidth);
  const rows = Math.max(1, Math.ceil(targetCards / columns));
  return Math.min(100, columns * rows);
}

function viewPageSize(kind) {
  if (kind === "cameras") {
    if (state.views.cameras === "list") return CAMERA_LIST_PAGE_SIZE;
    if (state.views.cameras === "dense") {
      return fullRowPageSize("cameraGrid", 150, CAMERA_DENSE_PAGE_SIZE);
    }
    return fullRowPageSize("cameraGrid", 248, CAMERA_PAGE_SIZE);
  }
  if (kind === "captures") {
    if (state.views.captures === "list") return CAPTURE_LIST_PAGE_SIZE;
    if (state.views.captures === "dense") {
      return fullRowPageSize("captureGrid", 150, CAPTURE_DENSE_PAGE_SIZE);
    }
    return fullRowPageSize("captureGrid", 240, CAPTURE_PAGE_SIZE);
  }
  if (state.views.review === "list") return REVIEW_LIST_PAGE_SIZE;
  if (state.views.review === "dense") {
    return fullRowPageSize("reviewList", 210, REVIEW_DENSE_PAGE_SIZE);
  }
  const reviewColumns = window.innerWidth <= 980 ? 1 : 2;
  return reviewColumns * Math.ceil(REVIEW_PAGE_SIZE / reviewColumns);
}

function applyView(kind) {
  const page = document.getElementById(`page-${kind}`);
  if (!page) return;
  const view = state.views[kind];
  page.dataset.view = view;
  document.querySelectorAll(`[data-view-for="${kind}"] button`).forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function scrollToPageTop() {
  window.scrollTo(0, 0);
  document.querySelector(".workspace")?.scrollTo(0, 0);
}

function setView(kind, view) {
  state.views[kind] = ["list", "dense"].includes(view) ? view : "grid";
  try {
    localStorage.setItem(`camangle.view.${kind}`, state.views[kind]);
  } catch (err) {
    // 浏览器禁用本地存储时仍保留当前页面会话内的选择。
  }
  applyView(kind);
  resetPaging();
  scrollToPageTop();
  loadPage(kind);
}

function debounceSearch(key, callback) {
  clearTimeout(state.searchTimers[key]);
  state.searchTimers[key] = setTimeout(callback, 260);
}

async function syncLatestBatch() {
  let progress = null;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    progress = await api("/api/cycle/progress");
    if (progress.batch_id !== null) {
      await loadBatches(Number(progress.batch_id));
      return progress;
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  await loadBatches(null);
  return progress;
}

function setStartCycleState(mode) {
  const btn = document.getElementById("startCycle");
  if (!btn) return;
  const states = {
    idle: { icon: "▶", title: "抓取最新画面", detail: "重新调用设备抓帧", disabled: false },
    starting: { icon: "◌", title: "正在启动", detail: "准备调用设备…", disabled: true },
    running: { icon: "◉", title: "正在抓取最新画面", detail: "本轮任务处理中", disabled: true },
  };
  const view = states[mode] || states.idle;
  btn.disabled = view.disabled;
  btn.dataset.state = mode;
  btn.querySelector(".start-cycle-icon").textContent = view.icon;
  btn.querySelector(".start-cycle-copy strong").textContent = view.title;
  btn.querySelector(".start-cycle-copy small").textContent = view.detail;
}

function refreshStartButton(p) {
  const running = Boolean(p && p.state === "running");
  setStartCycleState(running ? "running" : "idle");
  const sys = document.getElementById("sysState");
  if (sys) sys.textContent = running ? "任务运行中" : "系统在线";
}

function renderCaptureFlow(p) {
  const nodes = [...document.querySelectorAll("#captureFlow [data-flow-step]")];
  const steps = ["抓图", "下载", "分析", "入库"];
  const aliases = { "生成演示图": "分析" };
  const active = Array.isArray(p.active) ? p.active : [];
  const activeIndexes = active
    .map((task) => steps.indexOf(aliases[task.step] || task.step))
    .filter((index) => index >= 0);
  let current = -1;
  if (p.state === "finished") current = steps.length;
  if (p.state === "running") current = activeIndexes.length ? Math.min(...activeIndexes) : 0;

  nodes.forEach((node, index) => {
    node.classList.toggle("is-done", current > index);
    node.classList.toggle("is-active", p.state === "running" && current === index);
  });

  const flowState = document.getElementById("flowStateText");
  if (!flowState) return;
  if (p.state === "running") {
    flowState.textContent = active.length ? `${active.length} 路节点并行处理中` : "正在准备本轮任务";
  } else if (p.state === "finished") {
    flowState.textContent = `本轮完成 · ${p.processed || 0} 路最新画面`;
  } else if (p.state === "error") {
    flowState.textContent = p.error || "本轮处理异常";
  } else {
    flowState.textContent = "等待开始";
  }
}

function switchPage(page) {
  state.page = page;
  document.querySelectorAll(".side-nav button[data-page]").forEach((b) => {
    b.classList.toggle("active", b.dataset.page === page);
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("active", p.id === `page-${page}`);
  });
  scrollToPageTop();
  return loadPage(page);
}

async function loadPage(page) {
  if (page === "dashboard") await loadDashboard();
  if (page === "cameras") await loadCameras();
  if (page === "captures") await loadCaptures();
  if (page === "review") await loadReview();
  if (page === "settings") await loadSettings();
}

async function loadDashboard() {
  const statsQuery = withBatch(new URLSearchParams()).toString();
  const captureParams = withBatch(new URLSearchParams({ limit: String(recentRowLimit()) }));
  const [stats, captures, progress] = await Promise.all([
    api(`/api/stats${statsQuery ? `?${statsQuery}` : ""}`),
    api(`/api/captures?${captureParams.toString()}`),
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
  const p = progress || {};
  const batch = stats.batch || {};
  const liveBatch = p.batch_id !== null && Number(p.batch_id) === Number(batch.id);
  const cycleState = liveBatch ? p.state : (batch.state || "idle");
  const total = Number(liveBatch ? p.total : batch.total || 0);
  const processed = Number(liveBatch ? p.processed : batch.processed || 0);
  const percent = total ? processed * 100 / total : 0;
  document.getElementById("cycleState").textContent = stateMap[cycleState] || "空闲";
  document.getElementById("cycleStateTag").textContent = (cycleState || "idle").toUpperCase();
  document.getElementById("cyclePercent").textContent = `${percent.toFixed(1)}%`;
  document.getElementById("cycleProgress").textContent = `${processed} / ${total}（${percent.toFixed(1)}%）`;
  document.getElementById("cycleBar").style.width = `${Math.max(0, Math.min(100, percent))}%`;
  document.getElementById("cycleRate").textContent = liveBatch && p.rate_per_sec ? `${p.rate_per_sec} 路/秒` : "-";
  document.getElementById("cycleEta").textContent = liveBatch && cycleState === "running" ? fmtDuration(p.eta_seconds) : "-";
  document.getElementById("cycleStarted").textContent = fmtTime(batch.started_at);
  document.getElementById("cycleFinished").textContent = fmtTime(batch.finished_at);
  document.getElementById("headCycleState").textContent = stateMap[cycleState] || "空闲";
  document.getElementById("headReview").textContent = by.review || 0;
  renderActiveTasks(liveBatch ? (p.active || []) : []);
  renderCaptureFlow({
    ...p,
    state: cycleState,
    total,
    processed,
    active: liveBatch ? (p.active || []) : [],
  });
  refreshStartButton(p);

  const recent = document.getElementById("recentList");
  const rows = captures.rows || [];
  recent.innerHTML = rows.length ? rows.map((capture, index) => {
    const camera = capture.camera || {};
    const meta = camera.meta || {};
    const title = camera.channel_name || camera.device_serial || `相机 ${capture.camera_id}`;
    const serial = camera.device_serial || "-";
    const channel = camera.channel_no || "-";
    const thumb = capture.thumb_url || capture.image_url;
    const confidence = Number.isFinite(capture.confidence) ? capture.confidence.toFixed(2) : "-";
    const decision = capture.decision || "pending";
    const reason = (capture.reasons || [])[0] || "画面分析正常";
    return `
      <article class="recent-card">
        <button class="recent-frame" type="button" data-open="${capture.id}" aria-label="查看 ${escapeHtml(title)} 的抓帧详情">
          ${thumb
            ? `<img src="${escapeHtml(thumb)}" alt="${escapeHtml(title)} 抓帧" loading="lazy" />`
            : `<span class="no-frame"><b>未获取到画面</b><small>${escapeHtml(reason)}</small></span>`}
          <span class="recent-state ${escapeHtml(decision)}">${labelMap[decision] || escapeHtml(decision)}</span>
          ${index === 0 ? '<span class="latest-badge">最新</span>' : ""}
        </button>
        <div class="recent-card-body">
          <div class="recent-card-main">
            <div class="recent-card-copy">
              <h3>${escapeHtml(title)}</h3>
              <p>${escapeHtml(meta.store_name || "未标注门店")} · ${escapeHtml(serial)} / 通道 ${escapeHtml(channel)}</p>
            </div>
            <span class="recent-score"><b>${confidence}</b><small>置信度</small></span>
          </div>
          <div class="recent-card-foot">
            <span>${fmtRelativeTime(capture.capture_time)}</span>
            <button class="recent-link" type="button" data-open="${capture.id}">查看详情 <span>→</span></button>
          </div>
        </div>
      </article>`;
  }).join("") : `
    <div class="recent-empty">
      <span>◫</span>
      <div><b>还没有抓帧记录</b><p>点击右上角“抓取最新画面”开始本轮实时抓图。</p></div>
    </div>`;
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

async function loadCameras(page = 1) {
  const grid = document.getElementById("cameraGrid");
  const pageSize = viewPageSize("cameras");
  grid.setAttribute("aria-busy", "true");
  grid.innerHTML = loadingCards(8);
  document.getElementById("cameraFilterSummary").textContent = "正在加载相机节点…";
  try {
    const params = withBatch(new URLSearchParams({
      status: "active",
      limit: String(pageSize),
      offset: String((page - 1) * pageSize),
    }));
    const query = document.getElementById("cameraSearch").value.trim();
    const decision = document.getElementById("cameraStatusFilter").value;
    if (query) params.set("query", query);
    if (decision) params.set("decision", decision);
    const data = await api(`/api/cameras?${params.toString()}`);
    state.cameraPage = page;
    state.cameraTotal = Number(data.total || 0);
    document.getElementById("cameraCountLabel").textContent = state.cameraTotal;
    document.getElementById("cameraFilterSummary").textContent =
      `${state.batchAll ? "全部历史" : "当前批次"}匹配 ${state.cameraTotal} 个节点 · 第 ${page} 页`;
    renderCameraCards(data.rows || []);
    renderPagination("cameraPagination", {
      total: state.cameraTotal,
      current: page,
      pageSize,
      unit: "个节点",
      onPage: (next) => loadCameras(next),
    });
  } catch (err) {
    grid.innerHTML = `<div class="empty-state"><span>!</span><div><b>相机节点加载失败</b><p>${escapeHtml(err.message)}</p></div></div>`;
    document.getElementById("cameraPagination").innerHTML = "";
    toast(`节点加载失败：${err.message}`);
  } finally {
    grid.removeAttribute("aria-busy");
  }
}

function renderCameraCards(rows) {
  const grid = document.getElementById("cameraGrid");
  grid.innerHTML = rows.length ? rows.map((camera) => {
    const meta = camera.meta || {};
    const last = camera.last_capture;
    const decision = last ? last.decision : "pending";
    const decisionLabel = labelMap[decision] || "待处理";
    const channel = camera.channel_no ?? "-";
    const title = camera.channel_name || camera.device_serial || `相机 ${camera.id}`;
    const confidence = last && Number.isFinite(last.confidence) ? last.confidence.toFixed(2) : "-";
    const storeName = meta.store_name || "未标注门店";
    const storeNo = meta.store_no || "";
    const storeQuery = storeNo || storeName;
    return `
      <article
        class="node-card"
        data-state="${escapeHtml(decision)}"
        data-store-query="${escapeHtml(storeQuery)}"
        data-store-name="${escapeHtml(storeName)}"
        data-store-no="${escapeHtml(storeNo)}"
        role="button"
        tabindex="0"
        aria-label="查看 ${escapeHtml(storeName)} 的门店巡检汇总"
      >
        <div class="node-card-top">
          <span class="node-state ${escapeHtml(decision)}"><i></i>${decisionLabel}</span>
          <span class="node-channel">CH ${escapeHtml(channel)}</span>
        </div>
        <h2>${escapeHtml(title)}</h2>
        <p class="node-store">${escapeHtml(storeName)}</p>
        <div class="node-details">
          <div><span>序列号</span><b>${escapeHtml(camera.device_serial || "-")}</b></div>
          <div><span>场景模板</span><b>${escapeHtml(camera.scene_template || "default")}</b></div>
        </div>
        <div class="node-card-foot">
          <span>${last ? `最近检测 ${fmtRelativeTime(last.capture_time)}` : "尚未产生抓图"}</span>
          <span class="node-score"><b>${confidence}</b><small>置信度</small></span>
        </div>
      </article>`;
  }).join("") : `
    <div class="empty-state">
      <span>⌕</span><div><b>没有匹配的相机节点</b><p>尝试更换关键词或状态筛选。</p></div>
    </div>`;
}

async function loadCaptures(page = 1) {
  const grid = document.getElementById("captureGrid");
  const pageSize = viewPageSize("captures");
  grid.setAttribute("aria-busy", "true");
  grid.innerHTML = loadingCards(8);
  document.getElementById("captureSummary").textContent = "正在加载最新抓图…";
  try {
    const params = withBatch(new URLSearchParams({
      limit: String(pageSize),
      offset: String((page - 1) * pageSize),
    }));
    const query = document.getElementById("captureSearch").value.trim();
    const decision = document.getElementById("captureFilter").value;
    if (query) params.set("query", query);
    if (decision) params.set("decision", decision);
    const statsParams = withBatch(new URLSearchParams()).toString();
    const [data, stats] = await Promise.all([
      api(`/api/captures?${params.toString()}`),
      api(`/api/stats${statsParams ? `?${statsParams}` : ""}`),
    ]);
    const countBy = stats.by_decision || {};
    state.capturePage = page;
    state.captureTotal = Number(data.total || 0);
    document.getElementById("captureSummary").textContent =
      `${state.batchAll ? "全部历史" : "当前批次"}共 ${stats.total_captures || 0} 条 · 合格 ${countBy.pass || 0} · ` +
      `不合格 ${countBy.fail || 0} · 待复核 ${countBy.review || 0} · 无信号 ${countBy.no_signal || 0}` +
      ((query || decision) ? ` · 当前筛选 ${state.captureTotal} 条` : "");
    renderCaptureCards(data.rows || []);
    renderPagination("capturePagination", {
      total: state.captureTotal,
      current: page,
      pageSize,
      unit: "条",
      onPage: (next) => loadCaptures(next),
    });
  } catch (err) {
    grid.innerHTML = `<div class="empty-state"><span>!</span><div><b>抓图记录加载失败</b><p>${escapeHtml(err.message)}</p></div></div>`;
    document.getElementById("capturePagination").innerHTML = "";
    toast(`抓图加载失败：${err.message}`);
  } finally {
    grid.removeAttribute("aria-busy");
  }
}

function renderCaptureCards(rows) {
  document.getElementById("captureGrid").innerHTML = rows.length ? rows.map((capture) => {
    const cam = capture.camera || {};
    const meta = cam.meta || {};
    const decision = capture.decision || "pending";
    const title = cam.channel_name || cam.device_serial || `相机 ${capture.camera_id}`;
    const thumb = capture.thumb_url || capture.image_url;
    const reason = (capture.reasons || []).join("；") || "AI 未返回异常说明";
    const confidence = Number.isFinite(capture.confidence) ? capture.confidence.toFixed(2) : "-";
    return `
      <article class="capture-card ${escapeHtml(decision)}">
        <button class="capture-thumb" type="button" data-open="${capture.id}" aria-label="查看 ${escapeHtml(title)} 抓图">
          ${thumb
            ? `<img src="${escapeHtml(thumb)}" alt="${escapeHtml(title)} 抓图" loading="lazy" />`
            : `<span class="capture-no-image"><b>未获取画面</b><small>${escapeHtml(reason)}</small></span>`}
          <span class="capture-state ${escapeHtml(decision)}">${labelMap[decision] || escapeHtml(decision)}</span>
          <span class="capture-time">${fmtRelativeTime(capture.capture_time)}</span>
        </button>
        <div class="capture-meta">
          <div class="capture-title-row">
            <div>
              <h3>${escapeHtml(title)}</h3>
              <p class="capture-location">${escapeHtml(meta.store_name || "未标注门店")}</p>
            </div>
            <span class="capture-confidence"><b>${confidence}</b><small>置信度</small></span>
          </div>
          <div class="capture-device-row">
            <span>${escapeHtml(cam.device_serial || "-")}</span>
            <span>CH ${escapeHtml(cam.channel_no ?? "-")}</span>
          </div>
          <p class="capture-reason" title="${escapeHtml(reason)}">${escapeHtml(reason)}</p>
          <div class="capture-actions">
            <span>${fmtTime(capture.capture_time)}</span>
            <button type="button" data-open="${capture.id}">查看详情 <span>→</span></button>
          </div>
        </div>
      </article>`;
  }).join("") : `
    <div class="empty-state">
      <span>▤</span><div><b>没有匹配的抓图记录</b><p>尝试更换关键词或判定筛选。</p></div>
    </div>`;
}

async function loadReview(page = 1) {
  const list = document.getElementById("reviewList");
  const pageSize = viewPageSize("review");
  list.setAttribute("aria-busy", "true");
  list.innerHTML = loadingCards(4);
  document.getElementById("reviewFilterSummary").textContent = "正在加载复核队列…";
  try {
    const params = withBatch(new URLSearchParams({
      decision: "review",
      limit: String(pageSize),
      offset: String((page - 1) * pageSize),
    }));
    const query = document.getElementById("reviewSearch").value.trim();
    if (query) params.set("query", query);
    const statsParams = withBatch(new URLSearchParams()).toString();
    const [data, stats] = await Promise.all([
      api(`/api/captures?${params.toString()}`),
      api(`/api/stats${statsParams ? `?${statsParams}` : ""}`),
    ]);
    state.reviewTotal = Number(stats.by_decision?.review || 0);
    state.reviewPage = page;
    document.getElementById("reviewQueueCount").textContent = state.reviewTotal;
    document.getElementById("reviewFilterSummary").textContent =
      query ? `匹配 ${data.total || 0} 项 · 批次队列共 ${state.reviewTotal} 项` :
        `${state.batchAll ? "全部历史" : "当前批次"}共 ${state.reviewTotal} 项等待人工判断`;
    renderReviewCards(data.rows || []);
    renderPagination("reviewPagination", {
      total: Number(data.total || 0),
      current: page,
      pageSize,
      unit: "项",
      onPage: (next) => loadReview(next),
    });
  } catch (err) {
    list.innerHTML = `<div class="empty-state"><span>!</span><div><b>复核队列加载失败</b><p>${escapeHtml(err.message)}</p></div></div>`;
    document.getElementById("reviewPagination").innerHTML = "";
    toast(`复核队列加载失败：${err.message}`);
  } finally {
    list.removeAttribute("aria-busy");
  }
}

function renderReviewCards(rows) {
  document.getElementById("reviewList").innerHTML = rows.length ? rows.map((capture) => {
    const camera = capture.camera || {};
    const meta = camera.meta || {};
    const title = camera.channel_name || camera.device_serial || `相机 ${capture.camera_id}`;
    const reasons = capture.reasons || [];
    const reason = reasons.join("；") || "AI 无法确定，请结合画面人工判断";
    const confidence = Number.isFinite(capture.confidence) ? capture.confidence.toFixed(2) : "-";
    const score = Math.max(4, Math.min(100, Number(capture.confidence || 0) * 100));
    const baseline = capture.baseline_url
      ? `<figure class="review-figure">
          <button class="review-image-button" type="button" data-open="${capture.id}" aria-label="查看 ${escapeHtml(title)} 的基准图">
            <img src="${escapeHtml(capture.baseline_url)}" alt="基准图" loading="lazy" />
          </button>
          <figcaption>基准图</figcaption>
        </figure>`
      : "";
    return `
      <article class="review-item">
        <div class="review-media ${baseline ? "has-baseline" : ""}">
          <figure class="review-figure">
            <button class="review-image-button" type="button" data-open="${capture.id}" aria-label="查看 ${escapeHtml(title)} 的当前抓图">
              ${capture.image_url
                ? `<img src="${escapeHtml(capture.image_url)}" alt="${escapeHtml(title)} 当前抓图" loading="lazy" />`
                : `<span class="review-image-empty">当前抓图不可用</span>`}
            </button>
            <figcaption>AI 当前画面</figcaption>
          </figure>
          ${baseline}
          <span class="review-state">待复核</span>
        </div>
        <div class="review-info">
          <div class="review-card-head">
            <div>
              <h3>${escapeHtml(title)}</h3>
              <p>${escapeHtml(meta.store_name || "未标注门店")}</p>
            </div>
            <div class="review-confidence">
              <span>AI 置信度</span>
              <b>${confidence}</b>
              <i style="--score:${score}%"></i>
            </div>
          </div>
          <p class="review-reason">${escapeHtml(reason)}</p>
          <div class="review-meta">
            <span><small>设备序列号</small><b>${escapeHtml(camera.device_serial || "-")}</b></span>
            <span><small>通道</small><b>CH ${escapeHtml(camera.channel_no ?? "-")}</b></span>
            <span><small>抓图时间</small><b>${fmtRelativeTime(capture.capture_time)}</b></span>
          </div>
          <div class="review-actions">
            <button class="btn good" type="button" data-review="${capture.id}" data-decision="pass">标为合格</button>
            <button class="btn bad" type="button" data-review="${capture.id}" data-decision="fail">标为不合格</button>
          </div>
        </div>
      </article>`;
  }).join("") : `
    <div class="empty-state">
      <span>◎</span><div><b>没有匹配的复核项</b><p>尝试清空搜索或等待下一轮巡检。</p></div>
    </div>`;
}

async function submitReview(id, decision, setBaseline) {
  try {
    await api(`/api/captures/${id}/review`, {
      method: "POST",
      body: JSON.stringify({ decision, set_baseline: setBaseline, note: "" }),
    });
    toast("复核已提交");
    await loadReview();
    await loadBatches(currentBatchSelection());
    if (state.page === "dashboard") await loadDashboard();
    return true;
  } catch (err) {
    toast(`操作失败：${err.message}`);
    return false;
  }
}

function visibleCaseIds(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return [];
  const seen = new Set();
  const ids = [];
  container.querySelectorAll("[data-open]").forEach((element) => {
    const id = Number(element.dataset.open);
    if (!Number.isInteger(id) || seen.has(id)) return;
    seen.add(id);
    ids.push(id);
  });
  return ids;
}

function updateDetailNavigationState() {
  const total = state.detailIds.length;
  const index = state.detailIndex;
  const position = document.getElementById("detailNavPosition");
  const hint = document.getElementById("detailNavHint");
  const previous = document.getElementById("detailPrev");
  const next = document.getElementById("detailNext");
  if (position) position.textContent = total ? `第 ${index + 1} / ${total} 条` : "0 条";
  if (hint) {
    hint.textContent = total > 1
      ? `正在按当前筛选结果浏览第 ${index + 1} 条抓图，键盘 ← → 也可以切换。`
      : "当前列表只有一条抓图，切换筛选后可以连续查看其他记录。";
  }
  if (previous) previous.disabled = !total || index <= 0;
  if (next) next.disabled = !total || index >= total - 1;
}

async function navigateDetail(delta) {
  const nextIndex = state.detailIndex + delta;
  if (nextIndex < 0 || nextIndex >= state.detailIds.length) return;
  const ids = [...state.detailIds];
  try {
    await openDetail(ids[nextIndex], ids);
  } catch (err) {
    toast(`案例加载失败：${err.message}`);
    updateDetailNavigationState();
  }
}

async function openDetail(id, navigationIds = null) {
  const candidateIds = Array.isArray(navigationIds)
    ? [...new Set(navigationIds.map(Number).filter(Number.isInteger))]
    : [...state.detailIds];
  state.detailIds = candidateIds.includes(id) ? candidateIds : [id];
  state.detailIndex = state.detailIds.indexOf(id);
  updateDetailNavigationState();
  const requestId = ++state.detailRequestId;
  const detail = await api(`/api/captures/${id}`);
  if (requestId !== state.detailRequestId) return;
  document.getElementById("detailImage").src = detail.image_url;
  document.getElementById("detailBaseline").src = detail.baseline_url || EMPTY_IMG;
  document.getElementById("detailBaseline").closest("figure").hidden = !detail.baseline_url;
  document.querySelector("#detailModal .compare").style.gridTemplateColumns = detail.baseline_url ? "" : "1fr";
  const occlusion = detail.module_results && detail.module_results.occlusion;
  const mask = document.getElementById("detailOcclusionMask");
  const maskToggle = document.getElementById("detailOcclusionToggle");
  const maskToggleWrap = document.getElementById("detailOcclusionToggleWrap");
  const maskError = document.getElementById("detailOcclusionMaskError");
  mask.hidden = true;
  mask.removeAttribute("src");
  maskToggle.checked = true;
  maskToggleWrap.hidden = !detail.occlusion_url;
  maskError.hidden = true;
  mask.onload = () => { mask.hidden = !maskToggle.checked; };
  mask.onerror = () => {
    mask.hidden = true;
    maskError.hidden = false;
    maskToggleWrap.hidden = true;
  };
  if (detail.occlusion_url) mask.src = detail.occlusion_url;
  maskToggle.onchange = () => { mask.hidden = !maskToggle.checked; };
  document.getElementById("detailInfo").innerHTML = `
    <h3>判定详情</h3>
    <p>判定：${pill(detail.decision)} 置信度：${(detail.confidence || 0).toFixed(2)}</p>
    <p>原因：${escapeHtml((detail.reasons || []).join("；"))}</p>
    ${renderOcclusion(occlusion)}
    ${renderBlur(detail.module_results && detail.module_results.blur)}
    ${renderSemantic(detail.module_results && detail.module_results.semantic, detail.module_results && detail.module_results.element_match)}
    <details class="raw-metrics"><summary>查看全部图像指标</summary><p>${Object.entries(detail.metrics || {}).map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(typeof v === "object" ? JSON.stringify(v) : v)}`).join("， ")}</p></details>
    <div class="actions">
      <button class="btn good" data-review="${detail.id}" data-decision="pass">复核合格</button>
      <button class="btn bad" data-review="${detail.id}" data-decision="fail">复核不合格</button>
    </div>`;
  document.getElementById("detailModal").classList.remove("hidden");
  document.querySelector("#detailModal .modal-content").scrollTop = 0;
  updateDetailNavigationState();
  document.querySelectorAll("#detailInfo button[data-review]").forEach((b) => {
    b.addEventListener("click", async () => {
      const ok = await submitReview(Number(b.dataset.review), b.dataset.decision, b.dataset.baseline === "1");
      if (ok) document.getElementById("detailModal").classList.add("hidden");
    });
  });
}

function storeCameraDecision(camera) {
  return camera.last_capture?.decision || "pending";
}

function renderStoreSummary() {
  const counts = { pass: 0, fail: 0, review: 0, no_signal: 0, pending: 0 };
  state.storeRows.forEach((camera) => {
    const decision = storeCameraDecision(camera);
    if (Object.prototype.hasOwnProperty.call(counts, decision)) counts[decision] += 1;
  });
  const summary = document.getElementById("storeSummary");
  summary.innerHTML = `
    <div class="store-summary-card total"><span>通道总数</span><strong>${state.storeRows.length}</strong><small>当前门店</small></div>
    <div class="store-summary-card pass"><span>合格</span><strong>${counts.pass}</strong><small>画面检查通过</small></div>
    <div class="store-summary-card fail"><span>不合格</span><strong>${counts.fail}</strong><small>发现明确问题</small></div>
    <div class="store-summary-card review"><span>待复核</span><strong>${counts.review}</strong><small>AI 判断不确定</small></div>
    <div class="store-summary-card no-signal"><span>无信号</span><strong>${counts.no_signal}</strong><small>设备或抓图异常</small></div>
    <div class="store-summary-card pending"><span>待处理</span><strong>${counts.pending}</strong><small>当前批次尚无结果</small></div>`;
  const filterSummary = document.getElementById("storeFilterSummary");
  filterSummary.textContent =
    `${state.batchAll ? "全部历史最新结果" : "当前批次"} · 共 ${state.storeRows.length} 个通道 · 合格 ${counts.pass} / 不合格 ${counts.fail}`;
}

function renderStoreCameraList() {
  const list = document.getElementById("storeCameraList");
  const decisionOrder = { fail: 0, review: 1, no_signal: 2, pending: 3, pass: 4 };
  const rows = [...state.storeRows]
    .filter((camera) => !state.storeFilter || storeCameraDecision(camera) === state.storeFilter)
    .sort((a, b) => {
      const byDecision = (decisionOrder[storeCameraDecision(a)] ?? 9) - (decisionOrder[storeCameraDecision(b)] ?? 9);
      if (byDecision) return byDecision;
      return String(a.channel_name || a.device_serial || "").localeCompare(String(b.channel_name || b.device_serial || ""), "zh-CN");
    });
  list.innerHTML = rows.length ? rows.map((camera) => {
    const decision = storeCameraDecision(camera);
    const last = camera.last_capture;
    const title = camera.channel_name || camera.device_serial || `相机 ${camera.id}`;
    const confidence = last && Number.isFinite(last.confidence) ? last.confidence.toFixed(2) : "-";
    const captureId = last?.id;
    return `
      <article
        class="store-camera-row${captureId ? " is-openable" : ""}"
        data-state="${escapeHtml(decision)}"
        ${captureId ? `data-open="${captureId}" role="button" tabindex="0" aria-label="查看 ${escapeHtml(title)} 的抓图详情"` : ""}
      >
        <span class="store-camera-icon">CH</span>
        <div class="store-camera-copy">
          <strong>${escapeHtml(title)}</strong>
          <small>${escapeHtml(camera.device_serial || "-")} · 通道 ${escapeHtml(camera.channel_no ?? "-")}</small>
        </div>
        <span class="pill ${escapeHtml(decision)}">${labelMap[decision] || "待处理"}</span>
        <span class="store-camera-time">${last ? fmtRelativeTime(last.capture_time) : "尚未抓图"}</span>
        <span class="store-camera-score">${confidence}<small>置信度</small></span>
      </article>`;
  }).join("") : `
    <div class="empty-state store-empty">
      <span>⌕</span><div><b>该门店没有匹配的通道</b><p>尝试切换上方的判定筛选。</p></div>
    </div>`;
}

async function openStoreModal(storeQuery, storeName, storeNo = "") {
  state.storeName = storeName || "未标注门店";
  state.storeFilter = "";
  state.storeRows = [];
  document.getElementById("storeModalTitle").textContent = state.storeName;
  document.getElementById("storeModalMeta").textContent = "正在读取门店通道与当前判定…";
  document.getElementById("storeSummary").innerHTML = "";
  document.getElementById("storeCameraList").innerHTML = loadingCards(4);
  document.getElementById("storeFilterSummary").textContent = "正在加载…";
  document.querySelectorAll("#storeFilters button").forEach((button) => {
    const active = button.dataset.storeFilter === "";
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  document.getElementById("storeModal").classList.remove("hidden");
  try {
    const params = withBatch(new URLSearchParams({
      status: "active",
      limit: "100",
      query: storeQuery || state.storeName,
    }));
    const data = await api(`/api/cameras?${params.toString()}`);
    const targetNo = String(storeNo || "");
    state.storeRows = (data.rows || []).filter((camera) => {
      const meta = camera.meta || {};
      const rowName = String(meta.store_name || "");
      const rowNo = String(meta.store_no || "");
      return targetNo ? rowNo === targetNo : rowName === state.storeName;
    });
    if (!state.storeRows.length) {
      state.storeRows = data.rows || [];
    }
    const displayStoreNo = state.storeRows.find((camera) => camera.meta?.store_no)?.meta?.store_no;
    document.getElementById("storeModalMeta").textContent =
      `${state.batchAll ? "全部历史" : "当前批次"} · ${state.storeRows.length} 个通道${displayStoreNo ? ` · 门店号 ${displayStoreNo}` : ""}`;
    renderStoreSummary();
    renderStoreCameraList();
  } catch (err) {
    document.getElementById("storeModalMeta").textContent = "门店数据加载失败";
    document.getElementById("storeCameraList").innerHTML =
      `<div class="empty-state"><span>!</span><div><b>无法加载门店通道</b><p>${escapeHtml(err.message)}</p></div></div>`;
    toast(`门店数据加载失败：${err.message}`);
  }
}

function settingsFormSnapshot() {
  const snapshot = {};
  document.querySelectorAll("#page-settings input[id], #page-settings textarea[id]").forEach((input) => {
    if (input.type === "checkbox" || input.type === "radio") {
      snapshot[input.id] = input.checked;
    } else {
      snapshot[input.id] = input.value;
    }
  });
  return snapshot;
}

function applySettingsFormSnapshot(snapshot) {
  if (!snapshot) return;
  Object.entries(snapshot).forEach(([id, value]) => {
    const input = document.getElementById(id);
    if (!input) return;
    if (input.type === "checkbox" || input.type === "radio") input.checked = Boolean(value);
    else input.value = value;
  });
  updatePipelineModeUI();
  refreshPromptCounts();
}

function setSettingsDirty(dirty) {
  const stateEl = document.getElementById("settingsDirtyState");
  const title = document.getElementById("settingsSaveTitle");
  const subtitle = document.getElementById("settingsSaveSubtitle");
  const reset = document.getElementById("resetSettingsChanges");
  if (!stateEl) return;
  stateEl.classList.toggle("is-dirty", dirty);
  stateEl.querySelector("span").textContent = dirty ? "有未保存的修改" : "全部配置已保存";
  if (title) title.textContent = dirty ? "修改尚未保存" : "配置已同步";
  if (subtitle) {
    subtitle.textContent = dirty
      ? "保存后，运行参数、模块开关和 Prompt 将统一生效。"
      : "修改运行参数或 Prompt 后，在这里统一保存。";
  }
  if (reset) reset.disabled = !dirty;
}

function updateSettingsDirty() {
  if (!state.settingsSnapshot) return;
  const current = JSON.stringify(settingsFormSnapshot());
  const saved = JSON.stringify(state.settingsSnapshot);
  setSettingsDirty(current !== saved);
}

function refreshSettingsOverview() {
  const moduleSummary = document.getElementById("settingModuleSummary");
  const concurrencySummary = document.getElementById("settingConcurrencySummary");
  const intervalSummary = document.getElementById("settingIntervalSummary");
  const sampleSummary = document.getElementById("settingSampleSummary");
  const decoupled = document.getElementById("settingPipelineDecoupled")?.checked === true;
  if (moduleSummary) {
    moduleSummary.textContent = decoupled ? "先抓图后推理" : "抓图推理串联";
  }
  if (concurrencySummary) {
    const coupled = document.getElementById("settingConcurrency")?.value || "-";
    const capture = document.getElementById("settingCaptureConcurrency")?.value || "-";
    const inference = document.getElementById("settingInferenceConcurrency")?.value || "-";
    concurrencySummary.textContent = decoupled
      ? `${capture} 抓 / ${inference} 推`
      : `${coupled} 路`;
  }
  if (intervalSummary) {
    const value = document.getElementById("settingInterval")?.value;
    intervalSummary.textContent = value ? `${value} 秒` : "-";
  }
  if (sampleSummary) {
    const value = Number(document.getElementById("settingSampleSize")?.value || 0);
    sampleSummary.textContent = value > 0 ? `${value} 张随机` : "全部摄像头";
  }
}

function refreshPromptCounts() {
  const system = document.getElementById("settingPromptSystem");
  const user = document.getElementById("settingPromptUser");
  const systemCount = document.getElementById("systemPromptCount");
  const userCount = document.getElementById("userPromptCount");
  if (systemCount && system) systemCount.textContent = `${system.value.length} 字符`;
  if (userCount && user) userCount.textContent = `${user.value.length} 字符`;
}

function switchPromptTab(tab) {
  const activeTab = tab === "user" ? "user" : "system";
  document.querySelectorAll("#promptTabs [data-prompt-tab]").forEach((button) => {
    const active = button.dataset.promptTab === activeTab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".prompt-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.dataset.promptPane === activeTab);
  });
}

async function loadSettings() {
  const s = await api("/api/settings");
  const environment = s.demo_mode ? "演示环境" : "生产环境";
  document.getElementById("settingDemo").textContent = environment;
  document.getElementById("settingEnvironmentDetail").textContent = environment;
  document.getElementById("settingConcurrency").value = s.concurrency;
  document.getElementById("settingCaptureConcurrency").value = s.capture_concurrency;
  document.getElementById("settingInferenceConcurrency").value = s.inference_concurrency;
  document.getElementById("settingInterval").value = s.capture_interval;
  document.getElementById("settingSampleSize").value = s.sample_size ?? 0;
  document.getElementById("settingConfidence").value = s.thresholds.confidence_review;
  document.getElementById("settingRollFail").value = s.thresholds.roll_fail_deg;
  document.getElementById("settingBlurScore").value = s.thresholds.blur_var ?? 60;
  document.getElementById("settingBlurRatio").value = Number(((s.thresholds.blur_region_ratio ?? 0.5) * 100).toFixed(3));
  document.getElementById("settingOcclusionArea").value = Number(((s.thresholds.occlusion_area_ratio ?? 0.25) * 100).toFixed(3));
  document.getElementById("settingOcclusionDark").value = s.thresholds.occlusion_dark_value ?? 32;
  document.getElementById("settingOcclusionGray").value = s.thresholds.occlusion_gray_saturation ?? 35;
  document.getElementById("settingOcclusionTexture").value = s.thresholds.occlusion_texture_std ?? 10;
  document.getElementById("settingPipelineQwenOnly").checked = s.pipeline_mode !== "decoupled";
  document.getElementById("settingPipelineDecoupled").checked = s.pipeline_mode === "decoupled";
  state.serverSettings = s;
  document.getElementById("settingQwenAvailability").textContent = s.vlm_enabled
    ? `Qwen 服务已启用：${s.vlm_model}。两条链路使用同一套检测规则。`
    : "Qwen 服务当前未启用，两条检测链路均无法完成语义判断。";
  document.getElementById("settingPromptSystem").value = s.vlm_prompts?.system || "";
  document.getElementById("settingPromptUser").value = s.vlm_prompts?.user || "";
  const seedDemo = document.getElementById("seedDemo");
  seedDemo.disabled = !s.demo_mode;
  document.getElementById("seedDemoAvailability").textContent = s.demo_mode
    ? "当前处于演示环境，可以重新生成演示批次和图片。"
    : "当前是生产环境，演示数据重置已禁用，避免影响真实巡检数据。";
  updatePipelineModeUI();
  refreshPromptCounts();
  state.settingsSnapshot = settingsFormSnapshot();
  setSettingsDirty(false);
}

function updatePipelineModeUI() {
  document.querySelectorAll(".pipeline-mode-card").forEach((card) => {
    const input = card.querySelector("input[type=radio]");
    card.classList.toggle("enabled", Boolean(input && input.checked));
  });
  const decoupled = document.getElementById("settingPipelineDecoupled").checked;
  const coupledParams = document.getElementById("settingCoupledParams");
  const decoupledParams = document.getElementById("settingDecoupledParams");
  coupledParams.hidden = decoupled;
  decoupledParams.hidden = !decoupled;
  const coupledState = document.getElementById("settingPipelineCoupledState");
  const decoupledState = document.getElementById("settingPipelineDecoupledState");
  coupledState.textContent = decoupled ? "可选" : "当前使用";
  decoupledState.textContent = decoupled ? "当前使用" : "可选";
  const coupledConcurrency = document.getElementById("settingConcurrency").value || "-";
  const captureConcurrency = document.getElementById("settingCaptureConcurrency").value || "-";
  const inferenceConcurrency = document.getElementById("settingInferenceConcurrency").value || "-";
  document.getElementById("pipelineStageOne").textContent = decoupled ? "抓图 / 下载" : "抓图 + Qwen 推理";
  document.getElementById("pipelineStageOneMeta").textContent = decoupled
    ? `${captureConcurrency} 路并行`
    : `${coupledConcurrency} 路并行`;
  document.getElementById("pipelineStageTwo").textContent = decoupled ? "Qwen 批量推理" : "判定与入库";
  document.getElementById("pipelineStageTwoMeta").textContent = decoupled
    ? `${inferenceConcurrency} 张并行`
    : "每路任务完成后继续下一路";
  document.getElementById("settingPipelineImpactNote").textContent = decoupled
    ? "抓图阶段和推理阶段独立排队，适合用更高推理并发换取整体吞吐。"
    : "抓图和推理在同一任务内串行执行，配置简单，但同时在途任务更少。";
  const modules = state.serverSettings?.module_switches || {};
  const occlusion = modules.occlusion === true;
  const blur = modules.blur === true;
  const occlusionGroup = document.getElementById("occlusionSettings");
  const blurGroup = document.getElementById("blurSettings");
  const occlusionNote = document.getElementById("occlusionSettingsNote");
  const blurNote = document.getElementById("blurSettingsNote");
  occlusionGroup.classList.toggle("is-off", !occlusion);
  blurGroup.classList.toggle("is-off", !blur);
  occlusionNote.classList.toggle("is-off", !occlusion);
  blurNote.classList.toggle("is-off", !blur);
  occlusionNote.textContent = occlusion ? "已开启" : "已关闭";
  blurNote.textContent = blur ? "已开启" : "已关闭";
  occlusionNote.title = occlusion
    ? "模块已开启，阈值会参与下一轮巡检。"
    : "模块已关闭；阈值仍可修改，重新开启后生效。";
  blurNote.title = blur
    ? "模块已开启，阈值会参与下一轮巡检。"
    : "模块已关闭；阈值仍可修改，重新开启后生效。";
  refreshSettingsOverview();
}

function applyThresholdPreset(name) {
  const preset = THRESHOLD_PRESETS[name];
  if (!preset) return;
  Object.entries(preset).forEach(([id, value]) => {
    const input = document.getElementById(id);
    if (input) input.value = value;
  });
  updateSettingsDirty();
  toast(name === "occlusion" ? "已填入遮挡检测推荐阈值" : "已填入模糊检测推荐阈值");
}

async function saveSettingsNow(trigger) {
  const buttons = [...document.querySelectorAll("[data-save-settings]")];
  for (const input of document.querySelectorAll("#page-settings input[type=number]")) {
    if (!input.reportValidity()) {
      input.focus();
      toast("请先修正标红的配置项");
      return;
    }
  }
  const promptSystem = document.getElementById("settingPromptSystem").value.trim();
  const promptUser = document.getElementById("settingPromptUser").value.trim();
  if (!promptSystem || !promptUser) {
    toast("系统规则和任务 Prompt 都不能为空");
    return;
  }
  buttons.forEach((button) => {
    button.disabled = true;
    button.classList.add("is-saving");
    button.textContent = "保存中…";
  });
  try {
    await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify({
        thresholds: {
          confidence_review: Number(document.getElementById("settingConfidence").value),
          roll_fail_deg: Number(document.getElementById("settingRollFail").value),
          blur_var: Number(document.getElementById("settingBlurScore").value),
          blur_region_ratio: Number(document.getElementById("settingBlurRatio").value) / 100,
          occlusion_area_ratio: Number(document.getElementById("settingOcclusionArea").value) / 100,
          occlusion_dark_value: Number(document.getElementById("settingOcclusionDark").value),
          occlusion_gray_saturation: Number(document.getElementById("settingOcclusionGray").value),
          occlusion_texture_std: Number(document.getElementById("settingOcclusionTexture").value),
        },
        pipeline_mode: document.getElementById("settingPipelineDecoupled").checked
          ? "decoupled"
          : "coupled",
        vlm_prompts: {
          system: promptSystem,
          user: promptUser,
        },
        concurrency: Number(document.getElementById("settingConcurrency").value),
        capture_concurrency: Number(document.getElementById("settingCaptureConcurrency").value),
        inference_concurrency: Number(document.getElementById("settingInferenceConcurrency").value),
        capture_interval: Number(document.getElementById("settingInterval").value),
        sample_size: Number(document.getElementById("settingSampleSize").value),
      }),
    });
    state.settingsSnapshot = settingsFormSnapshot();
    setSettingsDirty(false);
    toast("设置已保存");
  } catch (err) {
    toast(`保存失败：${err.message}`);
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
      button.classList.remove("is-saving");
      button.textContent = "保存全部修改";
    });
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  Object.keys(state.views).forEach(applyView);
  document.querySelectorAll(".side-nav button[data-page]").forEach((b) => {
    b.addEventListener("click", () => switchPage(b.dataset.page));
  });
  document.querySelectorAll(".view-switch").forEach((switcher) => {
    switcher.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-view]");
      if (!button) return;
      const kind = switcher.dataset.viewFor;
      if (state.views[kind] === button.dataset.view) return;
      setView(kind, button.dataset.view);
    });
  });
  document.querySelectorAll("[data-close]").forEach((el) => {
    el.addEventListener("click", () => {
      document.getElementById(el.dataset.close).classList.add("hidden");
    });
  });
  document.getElementById("detailPrev").addEventListener("click", () => navigateDetail(-1));
  document.getElementById("detailNext").addEventListener("click", () => navigateDetail(1));
  document.getElementById("batchSelectButton").addEventListener("click", () => {
    const menu = document.getElementById("batchSelectMenu");
    setBatchMenuOpen(menu.hidden);
  });
  document.getElementById("batchViewAll").addEventListener("click", () => {
    const menu = document.getElementById("batchSelectMenu");
    setBatchMenuOpen(menu.hidden);
  });
  document.getElementById("batchSelectOptions").addEventListener("click", (event) => {
    const option = event.target.closest("[data-batch-scope]");
    if (!option) return;
    if (option.dataset.batchScope === "all") chooseBatch("all");
    else chooseBatch(Number(option.dataset.batchId));
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".batch-select-wrap, #batchViewAll")) closeBatchMenu();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      const detailModal = document.getElementById("detailModal");
      if (!detailModal.classList.contains("hidden")) {
        detailModal.classList.add("hidden");
        return;
      }
      const storeModal = document.getElementById("storeModal");
      if (!storeModal.classList.contains("hidden")) {
        storeModal.classList.add("hidden");
        return;
      }
      closeBatchMenu();
      document.getElementById("batchSelectButton").focus();
      return;
    }
    const detailModal = document.getElementById("detailModal");
    if (detailModal.classList.contains("hidden")) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName) || event.target.isContentEditable) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      navigateDetail(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      navigateDetail(1);
    }
  });
  document.getElementById("refreshBatches").addEventListener("click", async () => {
    await loadBatches(currentBatchSelection());
    await loadPage(state.page);
    toast("批次列表已刷新");
  });
  ["settingPipelineQwenOnly", "settingPipelineDecoupled"].forEach((id) => {
    document.getElementById(id).addEventListener("change", updatePipelineModeUI);
  });
  document.getElementById("page-settings").addEventListener("input", () => {
    updatePipelineModeUI();
    refreshSettingsOverview();
    refreshPromptCounts();
    updateSettingsDirty();
  });
  document.getElementById("page-settings").addEventListener("change", () => {
    updatePipelineModeUI();
    refreshSettingsOverview();
    updateSettingsDirty();
  });
  document.querySelectorAll("[data-threshold-preset]").forEach((button) => {
    button.addEventListener("click", () => applyThresholdPreset(button.dataset.thresholdPreset));
  });
  document.addEventListener("keydown", (event) => {
    if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "s") return;
    if (!document.getElementById("page-settings").classList.contains("active")) return;
    event.preventDefault();
    saveSettingsNow();
  });
  document.getElementById("promptTabs").addEventListener("click", (event) => {
    const tab = event.target.closest("[data-prompt-tab]");
    if (tab) switchPromptTab(tab.dataset.promptTab);
  });
  const activateSettingsSection = (sectionId) => {
    const target = document.getElementById(sectionId);
    if (!target) return;
    document.querySelectorAll(".settings-anchor").forEach((section) => {
      section.hidden = section !== target;
    });
    document.querySelectorAll(".settings-toc button").forEach((button) => {
      const active = button.dataset.settingsTarget === sectionId;
      button.classList.toggle("active", active);
      button.setAttribute("aria-expanded", String(active));
    });
    requestAnimationFrame(() => target.scrollIntoView({ behavior: "smooth", block: "start" }));
  };
  document.querySelector(".settings-toc").addEventListener("click", (event) => {
    const button = event.target.closest("[data-settings-target]");
    if (button) activateSettingsSection(button.dataset.settingsTarget);
  });
  document.querySelectorAll("[data-save-settings]").forEach((button) => {
    button.addEventListener("click", () => saveSettingsNow(button));
  });
  document.getElementById("resetSettingsChanges").addEventListener("click", () => {
    applySettingsFormSnapshot(state.settingsSnapshot);
    updateSettingsDirty();
    toast("已恢复为上次保存的配置");
  });
  document.getElementById("cameraSearch").addEventListener("input", () => {
    debounceSearch("camera", () => loadCameras(1));
  });
  document.getElementById("cameraStatusFilter").addEventListener("change", () => loadCameras(1));
  document.getElementById("cameraGrid").addEventListener("click", (event) => {
    const card = event.target.closest(".node-card[data-store-query]");
    if (card) openStoreModal(card.dataset.storeQuery, card.dataset.storeName, card.dataset.storeNo);
  });
  document.getElementById("cameraGrid").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const card = event.target.closest(".node-card[data-store-query]");
    if (!card) return;
    event.preventDefault();
    openStoreModal(card.dataset.storeQuery, card.dataset.storeName, card.dataset.storeNo);
  });
  document.getElementById("storeFilters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-store-filter]");
    if (!button) return;
    state.storeFilter = button.dataset.storeFilter;
    document.querySelectorAll("#storeFilters button").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    renderStoreCameraList();
  });
  document.getElementById("storeCameraList").addEventListener("click", (event) => {
    const row = event.target.closest(".store-camera-row[data-open]");
    if (row) openDetail(Number(row.dataset.open), visibleCaseIds("storeCameraList"));
  });
  document.getElementById("storeCameraList").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest(".store-camera-row[data-open]");
    if (!row) return;
    event.preventDefault();
    openDetail(Number(row.dataset.open), visibleCaseIds("storeCameraList"));
  });
  document.getElementById("captureSearch").addEventListener("input", () => {
    debounceSearch("captures", () => loadCaptures(1));
  });
  document.getElementById("captureFilter").addEventListener("change", () => loadCaptures(1));
  document.getElementById("reviewSearch").addEventListener("input", () => {
    debounceSearch("review", () => loadReview(1));
  });
  document.getElementById("captureGrid").addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-open]");
    if (trigger) openDetail(Number(trigger.dataset.open), visibleCaseIds("captureGrid"));
  });
  document.getElementById("reviewList").addEventListener("click", async (event) => {
    const reviewButton = event.target.closest("[data-review]");
    if (reviewButton) {
      const original = reviewButton.textContent;
      reviewButton.disabled = true;
      reviewButton.textContent = "提交中…";
      const ok = await submitReview(Number(reviewButton.dataset.review), reviewButton.dataset.decision, false);
      if (!ok) {
        reviewButton.disabled = false;
        reviewButton.textContent = original;
      }
      return;
    }
    const detailButton = event.target.closest("[data-open]");
    if (detailButton) openDetail(Number(detailButton.dataset.open), visibleCaseIds("reviewList"));
  });
  document.querySelector(".summary-strip").addEventListener("click", (event) => {
    const card = event.target.closest("[data-goto-page]");
    if (!card) return;
    if (card.dataset.gotoPage === "captures") {
      document.getElementById("captureFilter").value = card.dataset.decision || "";
    }
    switchPage(card.dataset.gotoPage);
  });
  document.getElementById("recentList").addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-open]");
    if (trigger) openDetail(Number(trigger.dataset.open), visibleCaseIds("recentList"));
  });
  document.getElementById("viewAllCaptures").addEventListener("click", () => switchPage("captures"));
  document.getElementById("startCycle").addEventListener("click", async () => {
    setStartCycleState("starting");
    try {
      const result = await api("/api/cycle/start", { method: "POST" });
      if (result.status === "already_running") {
        toast("本轮巡检仍在进行中");
        setStartCycleState("running");
      } else {
        toast("已开始抓取设备最新画面");
        setStartCycleState("running");
      }
      setTimeout(async () => {
        await syncLatestBatch();
        resetPaging();
        await loadPage(state.page);
      }, 500);
    } catch (err) {
      toast(`启动失败：${err.message}`);
      setStartCycleState("idle");
    }
  });
  document.getElementById("seedDemo").addEventListener("click", async () => {
    if (!confirm("重置演示数据会清空演示环境，确认继续？")) return;
    try {
      await api("/api/demo/seed", { method: "POST" });
      toast("演示数据已刷新");
      await loadBatches(null);
      await loadPage(state.page);
    } catch (err) {
      toast(`操作失败：${err.message}`);
    }
  });

  try {
    await loadBatches();
  } catch (err) {
    toast(`批次加载失败：${err.message}`);
  }
  await switchPage("dashboard");
  setInterval(() => {
    if (state.page === "dashboard") loadDashboard();
  }, 2000);
  setInterval(async () => {
    try {
      await loadBatches(currentBatchSelection());
    } catch (err) {
      // 批次栏刷新失败不阻塞主页面。
    }
  }, 5000);
  setInterval(async () => {
    try {
      const p = await api("/api/cycle/progress");
      refreshStartButton(p);
    } catch (err) {
      // 忽略瞬时错误，下一次轮询会再试。
    }
  }, 3000);
});
