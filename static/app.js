document.addEventListener('DOMContentLoaded', () => {
  const maxChars = 10000;
  const input = document.getElementById('emailInput');
  const senderInput = document.getElementById('senderInput');
  const btn = document.getElementById('predictBtn');
  const btnText = document.getElementById('btnText');
  const spinner = document.getElementById('spinner');
  const charCount = document.getElementById('charCount');
  const result = document.getElementById('result');
  const resultLabel = document.getElementById('resultLabel');
  const probBar = document.getElementById('probBar');
  const probText = document.getElementById('probText');
  const clearBtn = document.getElementById('clearBtn');
  const fieldsWarning = document.getElementById('fieldsWarning');
  const warningText = document.getElementById('warningText');
  const modelSelect = document.getElementById('modelSelect');

  const requiredMeta = [
    { el: senderInput, label: 'Sender' },
  ];

  function getMissingFields(){
    return requiredMeta
      .filter(f => f.el.value.trim().length === 0)
      .map(f => f.label);
  }

  function updateWarning(){
    const missing = getMissingFields();
    if (missing.length === 0){
      fieldsWarning.classList.add('hidden');
    } else {
      const names = missing.map(m => `<strong>${m}</strong>`);
      warningText.innerHTML = `Fill in ${names.join(' and ')} for accurate prediction.`;
      fieldsWarning.classList.remove('hidden');
    }
  }

  async function loadModels(){
    try{
      const resp = await fetch('/models');
      if (!resp.ok) return;
      const data = await resp.json();
      const models = Array.isArray(data) ? data : (data.models || []);
      if (!Array.isArray(models) || models.length === 0) return;
      modelSelect.innerHTML = '<option value="">Modèle par défaut</option>';
      models.forEach(m => {
        const name = typeof m === 'string' ? m : (m.name || m.id || String(m));
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        modelSelect.appendChild(opt);
      });
    }catch(e){
      // Silently ignore — default option remains
    }
  }

  function updateState(){
    const len = input.value.trim().length;
    const hasBody = len > 0;
    charCount.textContent = `${len} / ${maxChars}`;
    // Require body + sender
    const missing = getMissingFields();
    btn.disabled = !hasBody || missing.length > 0;
    updateWarning();
  }

  input.addEventListener('input', () => {
    if (input.value.length > maxChars) input.value = input.value.slice(0, maxChars);
    updateState();
  });

  // Listen on meta fields too — update warning + button state on every change
  senderInput.addEventListener('input', updateState);

  input.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!btn.disabled) btn.click();
    }
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    senderInput.value = '';
    updateState();
    result.classList.add('hidden');
    document.querySelectorAll('.example-btn').forEach(b => b.classList.remove('active'));
  });

  // example templates
  const exampleBtns = document.querySelectorAll('.example-btn');
  exampleBtns.forEach(btnEl => {
    btnEl.addEventListener('click', () => {
      input.value = btnEl.dataset.sample || '';
      senderInput.value = btnEl.dataset.sender || '';
      exampleBtns.forEach(b => b.classList.remove('active'));
      btnEl.classList.add('active');
      updateState();
      input.focus();
      input.scrollIntoView({behavior:'smooth', block:'center'});
    });
  });

  btn.addEventListener('click', async () => {
    const text = input.value.trim();
    if (!text) return;
    const missing = getMissingFields();
    if (missing.length > 0) return; // button should already be disabled
    btn.disabled = true;
    btn.classList.add('loading');
    btnText.textContent = 'Predicting…';
    result.classList.add('hidden');
    try{
      const payload = {
        body: text,
        sender: senderInput.value.trim(),
      };
      const selectedModel = modelSelect ? modelSelect.value : '';
      if (selectedModel) payload.model = selectedModel;
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      if (!resp.ok){
        const err = await resp.json().catch(()=>({detail:resp.statusText}));
        throw new Error(err.detail || 'Server error');
      }
      const data = await resp.json();
      const pred = data.prediction;
      const probArr = data.probability;
      const label = pred === 1 ? 'Phishing' : 'Not phishing';
      resultLabel.textContent = label;
      resultLabel.className = 'result-label ' + (pred === 1 ? 'phish' : 'safe');
      let pct = null;
      if (Array.isArray(probArr) && typeof pred === 'number'){
        pct = Math.round((probArr[pred] ?? Math.max(...probArr)) * 10000) / 100;
      }
      if (pct === null){
        probBar.style.width = '0%';
        probText.textContent = '';
      } else {
        probBar.style.width = `${pct}%`;
        probText.textContent = `${pct}%`;
      }
      result.classList.remove('hidden');
    }catch(e){
      resultLabel.textContent = 'Error: ' + e.message;
      resultLabel.className = 'result-label error';
      probBar.style.width = '0%';
      probText.textContent = '';
      result.classList.remove('hidden');
    }finally{
      btn.classList.remove('loading');
      btnText.textContent = 'Predict';
      updateState();
    }
  });

  // init
  updateState();
  loadModels();
});
