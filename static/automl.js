document.addEventListener('DOMContentLoaded', () => {
  'use strict';

  // ============================================================
  // Hyperparameter configurations per model
  // ============================================================
  const MODEL_CONFIGS = {
    xgboost: {
      label: 'XGBoost',
      params: [
        { name: 'n_estimators',  min: 50,  max: 1000, default: 100, step: 10, desc: 'Number of boosting rounds.' },
        { name: 'learning_rate', min: 0.01, max: 1.0,  default: 0.1, step: 0.01, desc: 'Step size shrinkage to prevent overfitting.' },
        { name: 'max_depth',     min: 1,   max: 15,   default: 6,   step: 1, desc: 'Maximum depth of a tree.' },
        { name: 'reg_alpha',     min: 0,   max: 10,   default: 0,   step: 0.1, desc: 'L1 regularization on weights.' },
        { name: 'reg_lambda',    min: 0,   max: 10,   default: 1,   step: 0.1, desc: 'L2 regularization on weights.' },
      ]
    },
    catboost: {
      label: 'CatBoost',
      params: [
        { name: 'iterations',    min: 50,  max: 1000, default: 100, step: 10, desc: 'Number of trees to build.' },
        { name: 'learning_rate', min: 0.01, max: 1.0,  default: 0.1, step: 0.01, desc: 'Step size shrinkage to prevent overfitting.' },
        { name: 'depth',         min: 1,   max: 15,   default: 6,   step: 1, desc: 'Depth of each tree.' },
        { name: 'l2_leaf_reg',   min: 0,   max: 10,   default: 3,   step: 0.1, desc: 'Coefficient at the L2 regularization term.' },
      ]
    }
  };

  const TUNING_METHOD_MAP = {
    random: 'RandomSearch',
    grid: 'GridSearch',
    bayesian: 'Bayesian'
  };

  // ============================================================
  // DOM references
  // ============================================================
  const modelInputs = document.querySelectorAll('input[name="model"]');
  const hyperparamsContainer = document.getElementById('hyperparamsContainer');
  const tuningSection = document.getElementById('tuningMethod').closest('section.card');
  const numRunsRow = document.getElementById('numRunsRow');
  const tuningMethodSelect = document.getElementById('tuningMethod');
  const numRunsInput = document.getElementById('numRuns');
  const trainingOverlay = document.getElementById('trainingOverlay');
  const launchBtn = document.getElementById('launchBtn');
  const launchBtnText = document.getElementById('launchBtnText');
  const launchSpinner = document.getElementById('launchSpinner');
  const leaderboardTable = document.getElementById('leaderboardTable');

  // ============================================================
  // State
  // ============================================================
  let currentModel = 'xgboost';
  let gridSearchWarningEl = null;
  let leaderboardSort = { column: null, direction: 'desc' };
  let lastResultsData = null;

  // ============================================================
  // Helpers
  // ============================================================
  function fmtValue(value, step) {
    const isIntStep = Number.isInteger(step) || (step % 1 === 0);
    if (isIntStep) return String(Math.round(value));
    const stepStr = String(step);
    const decimals = stepStr.includes('.') ? stepStr.split('.')[1].length : 0;
    return Number(value).toFixed(decimals);
  }

  // ============================================================
  // Render hyperparameters for the selected model
  // ============================================================
  function renderHyperparams(modelKey) {
    hyperparamsContainer.innerHTML = '';
    const cfg = MODEL_CONFIGS[modelKey];
    if (!cfg) return;

    cfg.params.forEach(p => {
      const card = document.createElement('div');
      card.className = 'param-card';
      card.dataset.param = p.name;

      // Header: name + range toggle
      const header = document.createElement('div');
      header.className = 'param-header';

      const nameLabel = document.createElement('label');
      nameLabel.className = 'param-name';
      nameLabel.textContent = p.name;
      if (p.desc) {
        const tooltip = document.createElement('span');
        tooltip.className = 'param-tooltip';
        tooltip.textContent = 'i';
        tooltip.title = p.desc;
        nameLabel.appendChild(tooltip);
      }

      const rangeToggle = document.createElement('label');
      rangeToggle.className = 'range-toggle';
      rangeToggle.innerHTML = `
        <input type="checkbox" class="range-checkbox" data-param="${p.name}">
        <span>Range</span>
      `;

      header.appendChild(nameLabel);
      header.appendChild(rangeToggle);

      // Value slider row (default visible)
      const valueRow = document.createElement('div');
      valueRow.className = 'param-value-row';
      valueRow.innerHTML = `
        <input type="range" class="param-slider" id="param-${p.name}"
               min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}">
        <span class="param-value" id="val-${p.name}">${fmtValue(p.default, p.step)}</span>
      `;

      // Range row (hidden by default)
      const rangeRow = document.createElement('div');
      rangeRow.className = 'param-range-row hidden';
      rangeRow.dataset.param = p.name;
      rangeRow.innerHTML = `
        <div class="range-field">
          <label>Min</label>
          <input type="range" class="range-min" id="min-${p.name}"
                 min="${p.min}" max="${p.max}" step="${p.step}" value="${p.min}">
          <span class="range-val" id="val-min-${p.name}">${fmtValue(p.min, p.step)}</span>
        </div>
        <div class="range-field">
          <label>Max</label>
          <input type="range" class="range-max" id="max-${p.name}"
                 min="${p.min}" max="${p.max}" step="${p.step}" value="${p.max}">
          <span class="range-val" id="val-max-${p.name}">${fmtValue(p.max, p.step)}</span>
        </div>
        <div class="range-field">
          <label>Step</label>
          <input type="number" class="range-step" id="step-${p.name}"
                 min="0.001" step="any" value="${p.step}">
        </div>
      `;

      card.appendChild(header);
      card.appendChild(valueRow);
      card.appendChild(rangeRow);
      hyperparamsContainer.appendChild(card);
    });

    updateTuningVisibility();
  }

  // ============================================================
  // Show/hide tuning section based on any active range mode
  // ============================================================
  function updateTuningVisibility() {
    const anyRange = hyperparamsContainer.querySelectorAll('.range-checkbox:checked').length > 0;
    if (anyRange) {
      tuningSection.classList.remove('hidden');
      numRunsRow.classList.remove('hidden');
    } else {
      tuningSection.classList.add('hidden');
      numRunsRow.classList.add('hidden');
    }
    updateGridSearchWarning();
  }

  // ============================================================
  // GridSearch run-count warning
  // ============================================================
  function updateGridSearchWarning() {
    const method = tuningMethodSelect.value;
    const nRuns = parseInt(numRunsInput.value, 10) || 0;

    if (gridSearchWarningEl) {
      gridSearchWarningEl.remove();
      gridSearchWarningEl = null;
    }

    if (method === 'grid' && nRuns > 50) {
      gridSearchWarningEl = document.createElement('div');
      gridSearchWarningEl.className = 'warning gridsearch-warning';
      gridSearchWarningEl.textContent =
        'Warning: GridSearch with more than 50 runs may be very slow.';
      numRunsRow.appendChild(gridSearchWarningEl);
    }
  }

  // ============================================================
  // Build JSON payload from current form state
  // ============================================================
  function buildPayload() {
    const cfg = MODEL_CONFIGS[currentModel];
    const params = {};
    const paramRanges = {};
    let hasRange = false;

    cfg.params.forEach(p => {
      const rangeCb = hyperparamsContainer.querySelector(
        `.range-checkbox[data-param="${p.name}"]`
      );

      if (rangeCb && rangeCb.checked) {
        hasRange = true;
        const minEl = document.getElementById(`min-${p.name}`);
        const maxEl = document.getElementById(`max-${p.name}`);
        const stepEl = document.getElementById(`step-${p.name}`);

        paramRanges[p.name] = {
          min: Number(minEl.value),
          max: Number(maxEl.value),
          step: Number(stepEl.value)
        };
      } else {
        const valEl = document.getElementById(`param-${p.name}`);
        params[p.name] = Number(valEl.value);
      }
    });

    const payload = {
      algo: cfg.label,
      params: params
    };

    if (hasRange) {
      const methodKey = tuningMethodSelect.value;
      const nRuns = parseInt(numRunsInput.value, 10) || 10;
      payload.tuning = {
        method: TUNING_METHOD_MAP[methodKey],
        n_runs: nRuns,
        param_ranges: paramRanges
      };
    }

    return payload;
  }

  // ============================================================
  // Form submission
  // ============================================================
  async function handleSubmit() {
    cancelPending();
    launchBtn.disabled = true;
    launchBtn.classList.add('loading');
    launchBtnText.textContent = 'Launching…';
    launchSpinner.classList.remove('hidden');

    try {
      const payload = buildPayload();
      currentAbortController = new AbortController();
      const resp = await fetch('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: currentAbortController.signal
      });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      let msg = 'Server error';
      if (Array.isArray(err.detail)) {
        msg = err.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
      } else if (typeof err.detail === 'string') {
        msg = err.detail;
      } else if (err.detail) {
        msg = JSON.stringify(err.detail);
      }
      throw new Error(msg);
    }

      const data = await resp.json();
      // Hand-off to Part 2 (progress / results) via custom event
      document.dispatchEvent(new CustomEvent('automl:training-started', { detail: data }));
      if (trainingOverlay) trainingOverlay.classList.remove('hidden');
    } catch (e) {
      if (e.name === 'AbortError') return;
      if (trainingOverlay) trainingOverlay.classList.add('hidden');
      alert('Training error: ' + e.message);
    } finally {
      launchBtn.classList.remove('loading');
      launchBtnText.textContent = "Lancer l'entraînement";
      launchSpinner.classList.add('hidden');
      launchBtn.disabled = false;
    }
  }

  // ============================================================
  // Event delegation for hyperparameter container
  // ============================================================
  hyperparamsContainer.addEventListener('input', (e) => {
    const target = e.target;

    // Value slider → update display
    if (target.classList.contains('param-slider')) {
      const valSpan = document.getElementById(target.id.replace('param-', 'val-'));
      const paramName = target.id.replace('param-', '');
      const cfg = MODEL_CONFIGS[currentModel].params.find(p => p.name === paramName);
      if (valSpan && cfg) valSpan.textContent = fmtValue(target.value, cfg.step);
      return;
    }

    // Range min slider → update display and clamp
    if (target.classList.contains('range-min')) {
      const paramName = target.id.replace('min-', '');
      const maxEl = document.getElementById(`max-${paramName}`);
      const valSpan = document.getElementById(target.id.replace('min-', 'val-min-'));
      const cfg = MODEL_CONFIGS[currentModel].params.find(p => p.name === paramName);
      let val = Number(target.value);
      if (maxEl && val > Number(maxEl.value)) {
        val = Number(maxEl.value);
        target.value = val;
      }
      if (valSpan && cfg) valSpan.textContent = fmtValue(val, cfg.step);
      return;
    }

    // Range max slider → update display and clamp
    if (target.classList.contains('range-max')) {
      const paramName = target.id.replace('max-', '');
      const minEl = document.getElementById(`min-${paramName}`);
      const valSpan = document.getElementById(target.id.replace('max-', 'val-max-'));
      const cfg = MODEL_CONFIGS[currentModel].params.find(p => p.name === paramName);
      let val = Number(target.value);
      if (minEl && val < Number(minEl.value)) {
        val = Number(minEl.value);
        target.value = val;
      }
      if (valSpan && cfg) valSpan.textContent = fmtValue(val, cfg.step);
      return;
    }
  });

  hyperparamsContainer.addEventListener('change', (e) => {
    const target = e.target;

    // Range checkbox → toggle range row visibility
    if (target.classList.contains('range-checkbox')) {
      const paramName = target.dataset.param;
      const card = hyperparamsContainer.querySelector(`.param-card[data-param="${paramName}"]`);
      if (!card) return;
      const rangeRow = card.querySelector('.param-range-row');
      const valueRow = card.querySelector('.param-value-row');

      if (target.checked) {
        rangeRow.classList.remove('hidden');
        valueRow.classList.add('hidden');
      } else {
        rangeRow.classList.add('hidden');
        valueRow.classList.remove('hidden');
      }
      updateTuningVisibility();
      return;
    }
  });

  // ============================================================
  // Direct listeners
  // ============================================================
  modelInputs.forEach(input => {
    input.addEventListener('change', () => {
      if (input.checked) {
        currentModel = input.value;
        renderHyperparams(currentModel);
      }
    });
  });

  tuningMethodSelect.addEventListener('change', updateGridSearchWarning);
  numRunsInput.addEventListener('input', updateGridSearchWarning);

  launchBtn.addEventListener('click', handleSubmit);

  // ============================================================
  // Init
  // ============================================================
  const checkedModel = document.querySelector('input[name="model"]:checked');
  currentModel = checkedModel ? checkedModel.value : 'xgboost';
  renderHyperparams(currentModel);

  // ============================================================
  // Progress polling & results display
  // ============================================================
  let currentJobId = null;
  let pollInterval = null;
  let currentAbortController = null;

  function cancelPending() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
    if (currentAbortController) {
      currentAbortController.abort();
      currentAbortController = null;
    }
  }

  function sortRuns(runs, column, direction) {
    if (!column) return runs.slice();
    const dir = direction === 'asc' ? 1 : -1;
    return runs.slice().sort((a, b) => {
      let av, bv;
      if (column === 'run_number') {
        av = a.run_number || 0;
        bv = b.run_number || 0;
      } else if (column === 'score') {
        av = a.score || 0;
        bv = b.score || 0;
      } else if (column === 'f1_score') {
        av = a.metrics?.f1_score || 0;
        bv = b.metrics?.f1_score || 0;
      } else if (column === 'roc_auc') {
        av = a.metrics?.roc_auc || 0;
        bv = b.metrics?.roc_auc || 0;
      } else {
        return 0;
      }
      return (av - bv) * dir;
    });
  }

  function updateSortIndicators() {
    if (!leaderboardTable) return;
    leaderboardTable.querySelectorAll('th.sortable').forEach(th => {
      th.classList.remove('sort-asc', 'sort-desc');
      if (th.dataset.sort === leaderboardSort.column) {
        th.classList.add(leaderboardSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
      }
    });
  }

  function renderResults(data) {
    if (!data) return;
    const resultsSection = document.getElementById('resultsSection');
    const bestAccuracy = document.getElementById('bestAccuracy');
    const bestF1 = document.getElementById('bestF1');
    const bestRocAuc = document.getElementById('bestRocAuc');
    const bestRun = document.getElementById('bestRun');
    const leaderboardBody = document.getElementById('leaderboardBody');

    resultsSection.classList.remove('hidden');

    const br = data.best_run || {};
    bestAccuracy.textContent = (br.score || 0).toFixed(4);
    bestF1.textContent = (br.metrics?.f1_score || 0).toFixed(4);
    bestRocAuc.textContent = (br.metrics?.roc_auc || 0).toFixed(4);
    bestRun.textContent = '#' + (br.run_number || 1);

    leaderboardBody.replaceChildren();
    const runs = Array.isArray(data.runs) ? data.runs : [];
    let displayRuns = sortRuns(runs, leaderboardSort.column, leaderboardSort.direction);
    const bestScore = displayRuns.length > 0 ? Math.max(...displayRuns.map(r => r.score || 0)) : -Infinity;

    displayRuns.forEach(run => {
      const tr = document.createElement('tr');
      const isBest = (run.score || 0) >= bestScore;
      if (isBest) tr.classList.add('best-run');

      const tdRun = document.createElement('td');
      tdRun.textContent = '#' + (run.run_number || 1);
      tr.appendChild(tdRun);

      const tdScore = document.createElement('td');
      tdScore.textContent = (run.score || 0).toFixed(4);
      tr.appendChild(tdScore);

      const tdF1 = document.createElement('td');
      tdF1.textContent = (run.metrics?.f1_score || 0).toFixed(4);
      tr.appendChild(tdF1);

      const tdRocAuc = document.createElement('td');
      tdRocAuc.textContent = (run.metrics?.roc_auc || 0).toFixed(4);
      tr.appendChild(tdRocAuc);

      const tdParams = document.createElement('td');
      tdParams.className = 'params-cell';
      tdParams.textContent = JSON.stringify(run.params);
      tr.appendChild(tdParams);

      leaderboardBody.appendChild(tr);
    });
  }

  function startPolling(jobId) {
    cancelPending();
    currentAbortController = new AbortController();
    currentJobId = jobId;
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressStatus = document.getElementById('progressStatus');
    const progressDetail = document.getElementById('progressDetail');

    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    progressBar.style.background = ''; // reset red if any

    pollInterval = setInterval(async () => {
      try {
        const resp = await fetch(`/train/${encodeURIComponent(jobId)}/status`, {
          signal: currentAbortController.signal
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        progressBar.style.width = data.progress + '%';
        progressPercent.textContent = data.progress + '%';
        progressStatus.textContent = data.message;
        progressDetail.textContent = (data.current_run || 0) + '/' + (data.total_runs || 0);

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          pollInterval = null;
          if (trainingOverlay) trainingOverlay.classList.add('hidden');
          fetchResults(jobId);
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          pollInterval = null;
          if (trainingOverlay) trainingOverlay.classList.add('hidden');
          progressStatus.textContent = 'Failed: ' + data.message;
          progressBar.style.background = '#ef4444';
        }
      } catch (e) {
        if (e.name === 'AbortError') {
          clearInterval(pollInterval);
          pollInterval = null;
          return;
        }
        clearInterval(pollInterval);
        pollInterval = null;
        if (trainingOverlay) trainingOverlay.classList.add('hidden');
        progressStatus.textContent = 'Error: ' + e.message;
        progressBar.style.background = '#ef4444';
      }
    }, 2500);
  }

  async function fetchResults(jobId) {
    try {
      const resp = await fetch(`/train/${encodeURIComponent(jobId)}/results`, {
        signal: currentAbortController.signal
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      lastResultsData = data;
      renderResults(data);
    } catch (e) {
      if (e.name === 'AbortError') return;
      alert('Failed to load results: ' + e.message);
    }
  }

  document.addEventListener('automl:training-started', (e) => {
    const jobId = e.detail.job_id;
    if (jobId) startPolling(jobId);
  });

  const saveModelBtn = document.getElementById('saveModelBtn');
  const downloadModelBtn = document.getElementById('downloadModelBtn');

  if (saveModelBtn) {
    saveModelBtn.addEventListener('click', async () => {
      if (!currentJobId) {
        alert('No training job to save.');
        return;
      }
      const name = prompt('Enter model name:');
      if (!name) return;
      try {
        const resp = await fetch(`/models/${encodeURIComponent(name)}/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_id: currentJobId })
        });
        if (!resp.ok) throw new Error(await resp.text());
        alert('Model saved successfully!');
      } catch (e) {
        alert('Save failed: ' + e.message);
      }
    });
  }

  if (downloadModelBtn) {
    downloadModelBtn.addEventListener('click', () => {
      const name = prompt('Enter model name to download:');
      if (!name) return;
      window.location.href = `/models/${encodeURIComponent(name)}/download`;
    });
  }

  if (leaderboardTable) {
    leaderboardTable.querySelectorAll('th.sortable').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.dataset.sort;
        if (leaderboardSort.column === col) {
          leaderboardSort.direction = leaderboardSort.direction === 'asc' ? 'desc' : 'asc';
        } else {
          leaderboardSort.column = col;
          leaderboardSort.direction = 'desc';
        }
        updateSortIndicators();
        if (lastResultsData) renderResults(lastResultsData);
      });
    });
  }
});
