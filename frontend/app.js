let modelContract = null;

function setSelectOptions(selectEl, values) {
  selectEl.innerHTML = "";
  for (const val of values) {
    const opt = document.createElement("option");
    opt.value = String(val);
    opt.textContent = String(val);
    selectEl.appendChild(opt);
  }
}

function updateDerivedDisplays() {
  if (!modelContract) return;
  const plan = document.getElementById("subscription_plan").value;
  const country = document.getElementById("country").value;
  const wt = parseFloat(document.getElementById("watch_time_minutes").value);
  const sc = parseFloat(document.getElementById("session_count").value);

  const fee = modelContract.plan_to_fee[plan];
  document.getElementById("derived_monthly_fee").value =
    fee !== undefined ? `$${Number(fee).toFixed(2)}` : "—";

  const region = modelContract.country_to_region[country];
  document.getElementById("derived_region").value = region || "—";

  if (Number.isFinite(wt) && Number.isFinite(sc) && sc >= 1) {
    document.getElementById("derived_wtps").value = `${(wt / sc).toFixed(2)} min/session`;
  } else {
    document.getElementById("derived_wtps").value = "Requires session_count >= 1";
  }
}

function applyPreset(presetKey) {
  if (!modelContract || !modelContract.synthetic_presets[presetKey]) return;
  const inputs = modelContract.synthetic_presets[presetKey].inputs;
  document.getElementById("tenureMode").value = "days";
  toggleTenureMode();

  for (const [key, val] of Object.entries(inputs)) {
    const el = document.getElementById(key);
    if (el) {
      el.value = String(val);
    }
  }
  updateDerivedDisplays();
  clearError();
}

function toggleTenureMode() {
  const mode = document.getElementById("tenureMode").value;
  const daysWrap = document.getElementById("tenureDaysWrap");
  const dateWrap = document.getElementById("startDateWrap");
  if (mode === "days") {
    daysWrap.classList.remove("hidden");
    dateWrap.classList.add("hidden");
  } else {
    daysWrap.classList.add("hidden");
    dateWrap.classList.remove("hidden");
  }
}

function showError(message, fieldErrors = null) {
  const banner = document.getElementById("errorBanner");
  let html = `<strong>${escapeHtml(message)}</strong>`;
  if (fieldErrors && typeof fieldErrors === "object") {
    const items = Object.entries(fieldErrors)
      .map(([k, v]) => `<li><code>${escapeHtml(k)}</code>: ${escapeHtml(v)}</li>`)
      .join("");
    html += `<ul>${items}</ul>`;
  }
  banner.innerHTML = html;
  banner.classList.remove("hidden");
}

function clearError() {
  const banner = document.getElementById("errorBanner");
  banner.classList.add("hidden");
  banner.innerHTML = "";
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderGlobalInsights(contract) {
  const wrap = document.getElementById("globalInsightsWrap");
  const permRows = contract.global_permutation_importance_top5
    .map(
      (r) =>
        `<tr><td><code>${escapeHtml(r.feature)}</code></td><td>+${Number(r.perm_f1_drop).toFixed(5)}</td></tr>`
    )
    .join("");

  const planRows = Object.entries(contract.train_plan_churn_rates)
    .map(
      ([plan, info]) =>
        `<tr><td>${escapeHtml(plan)}</td><td>${info.train_count.toLocaleString()}</td><td>${info.churn_rate_pct}%</td></tr>`
    )
    .join("");

  wrap.innerHTML = `
    <div class="subpanel">
      <h4>Global Permutation Importance (X_train Val F1 Drop)</h4>
      <table>
        <thead><tr><th>Feature</th><th>Mean F1 Drop</th></tr></thead>
        <tbody>${permRows}</tbody>
      </table>
    </div>
    <div class="subpanel">
      <h4>X_train Churn Rate by Subscription Plan</h4>
      <table>
        <thead><tr><th>Plan</th><th>X_train n</th><th>Churn Rate</th></tr></thead>
        <tbody>${planRows}</tbody>
      </table>
    </div>
  `;
}

function renderRiskBandTable(contract, activeBandName = null) {
  const rows = contract.risk_bands
    .map((b) => {
      const rangeStr =
        b.max_prob_exclusive > 1
          ? `&ge; ${(b.min_prob_inclusive * 100).toFixed(0)}%`
          : `${(b.min_prob_inclusive * 100).toFixed(0)}% – &lt;${(b.max_prob_exclusive * 100).toFixed(0)}%`;
      const cls = b.name === activeBandName ? "row-active-band" : "";
      return `
        <tr class="${cls}">
          <td><strong>${escapeHtml(b.name)}</strong></td>
          <td>${rangeStr}</td>
          <td>${b.observed_churn_rate_pct}% (n=${b.train_cv_count.toLocaleString()})</td>
          <td>${escapeHtml(b.basis)}</td>
        </tr>
      `;
    })
    .join("");

  document.getElementById("riskBandTable").innerHTML = `
    <table>
      <thead>
        <tr><th>Tier</th><th>Score Range</th><th>X_train CV Churn Rate</th><th>Threshold Basis</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderPredictionResult(data) {
  document.getElementById("resultEmpty").classList.add("hidden");
  document.getElementById("resultContent").classList.remove("hidden");
  document.getElementById("resultSubtitle").textContent =
    `Model: ${data.model_name} (${data.model_version})`;

  const pct = Number(data.estimated_churn_probability_pct);
  document.getElementById("probPct").textContent = `${pct.toFixed(2)}%`;
  document.getElementById("gaugeFill").style.width = `${Math.min(Math.max(pct, 0), 100)}%`;

  const badge = document.getElementById("riskBadge");
  badge.textContent = `${data.risk_category} Risk`;
  badge.className = "risk-badge";
  if (data.risk_category === "Low") badge.classList.add("risk-low");
  else if (data.risk_category === "Moderate") badge.classList.add("risk-moderate");
  else badge.classList.add("risk-high");

  document.getElementById("verdictChip").textContent =
    `Model binary label (50% cutoff): ${data.predicted_label}`;
  document.getElementById("probDisclaimer").textContent = data.probability_disclaimer;

  renderRiskBandTable(modelContract, data.risk_category);

  const d = data.derived_features;
  document.getElementById("derivedSummary").innerHTML =
    `<strong>Contract Features Sent to Pipeline (24 cols):</strong> ` +
    `monthly_fee=<code>$${Number(d.monthly_fee).toFixed(2)}</code>, ` +
    `region=<code>${escapeHtml(d.region)}</code>, ` +
    `tenure_days=<code>${d.tenure_days}</code>, ` +
    `watch_time_per_session=<code>${d.watch_time_per_session}</code>`;

  const ctxRows = data.descriptive_context
    .map(
      (r) => `
      <tr>
        <td>${escapeHtml(r.label)}</td>
        <td><strong>${r.submitted_value}</strong></td>
        <td>${r.train_active_mean}</td>
        <td>${r.train_cancelled_mean}</td>
      </tr>
    `
    )
    .join("");

  document.getElementById("descriptiveTableWrap").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Feature</th>
          <th>Submitted</th>
          <th>X_train Active Mean</th>
          <th>X_train Cancelled Mean</th>
        </tr>
      </thead>
      <tbody>${ctxRows}</tbody>
    </table>
  `;
}

async function initApp() {
  try {
    const resp = await fetch("/contract");
    const contract = await resp.json();
    if (!resp.ok) {
      document.getElementById("modelStatusBadge").textContent = "Model unavailable";
      showError(contract.error || "Failed to load model contract.");
      return;
    }

    modelContract = contract;
    const m = contract.verified_test_metrics;
    document.getElementById("modelStatusBadge").textContent =
      `${contract.model_name} • Test ROC-AUC ${m.roc_auc} | AP ${m.average_precision} | Recall ${(m.recall * 100).toFixed(1)}%`;

    for (const [col, vals] of Object.entries(contract.allowed_categories)) {
      const sel = document.getElementById(col);
      if (sel) setSelectOptions(sel, vals);
    }

    const presetWrap = document.getElementById("presetButtons");
    presetWrap.innerHTML = "";
    for (const [key, info] of Object.entries(contract.synthetic_presets)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-preset";
      btn.textContent = info.label;
      btn.title = info.note;
      btn.addEventListener("click", () => applyPreset(key));
      presetWrap.appendChild(btn);
    }

    renderGlobalInsights(contract);
    renderRiskBandTable(contract);
    applyPreset("moderate_risk_demo");
  } catch (err) {
    document.getElementById("modelStatusBadge").textContent = "Server connection failed";
    showError(`Could not reach backend API: ${err.message}`);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initApp();

  document.getElementById("tenureMode").addEventListener("change", toggleTenureMode);
  for (const id of ["subscription_plan", "country", "watch_time_minutes", "session_count"]) {
    document.getElementById(id).addEventListener("input", updateDerivedDisplays);
    document.getElementById(id).addEventListener("change", updateDerivedDisplays);
  }

  document.getElementById("resetBtn").addEventListener("click", () => {
    applyPreset("moderate_risk_demo");
  });

  document.getElementById("churnForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    const payload = {
      subscription_plan: document.getElementById("subscription_plan").value,
      country: document.getElementById("country").value,
      age_group: Number(document.getElementById("age_group").value),
      gender: document.getElementById("gender").value,
      payment_method: document.getElementById("payment_method").value,
      discount_applied: document.getElementById("discount_applied").value,
      content_type: document.getElementById("content_type").value,
      genre: document.getElementById("genre").value,
      language: document.getElementById("language").value,
      device_type: document.getElementById("device_type").value,
      time_of_day: document.getElementById("time_of_day").value,
      recommendation_source: document.getElementById("recommendation_source").value,
      rating: Number(document.getElementById("rating").value),
      liked: document.getElementById("liked").value,
      watch_time_minutes: Number(document.getElementById("watch_time_minutes").value),
      session_count: Number(document.getElementById("session_count").value),
      completion_percentage: Number(document.getElementById("completion_percentage").value),
      days_since_last_watch: Number(document.getElementById("days_since_last_watch").value),
      avg_weekly_watch_time: Number(document.getElementById("avg_weekly_watch_time").value),
      content_diversity_score: Number(document.getElementById("content_diversity_score").value),
    };

    const mode = document.getElementById("tenureMode").value;
    if (mode === "days") {
      payload.tenure_days = Number(document.getElementById("tenure_days").value);
    } else {
      payload.subscription_start_date = document.getElementById("subscription_start_date").value;
    }

    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) {
        showError(data.error || "Prediction request failed.", data.field_errors);
        return;
      }
      renderPredictionResult(data);
    } catch (err) {
      showError(`Network error calling /predict: ${err.message}`);
    }
  });
});

