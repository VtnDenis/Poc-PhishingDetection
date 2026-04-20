document.addEventListener('DOMContentLoaded', () => {
  const maxChars = 10000;
  const input = document.getElementById('emailInput');
  const btn = document.getElementById('predictBtn');
  const btnText = document.getElementById('btnText');
  const spinner = document.getElementById('spinner');
  const charCount = document.getElementById('charCount');
  const result = document.getElementById('result');
  const resultLabel = document.getElementById('resultLabel');
  const probBar = document.getElementById('probBar');
  const probText = document.getElementById('probText');
  const clearBtn = document.getElementById('clearBtn');

  function updateState(){
    const len = input.value.trim().length;
    charCount.textContent = `${len} / ${maxChars}`;
    btn.disabled = len === 0;
  }

  input.addEventListener('input', () => {
    if (input.value.length > maxChars) input.value = input.value.slice(0, maxChars);
    updateState();
  });

  input.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!btn.disabled) btn.click();
    }
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    updateState();
    result.classList.add('hidden');
    // remove active state from example buttons
    document.querySelectorAll('.example-btn').forEach(b => b.classList.remove('active'));
  });

  // example templates
  const exampleBtns = document.querySelectorAll('.example-btn');
  exampleBtns.forEach(btnEl => {
    btnEl.addEventListener('click', () => {
      input.value = btnEl.dataset.sample || '';
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
    btn.disabled = true;
    btn.classList.add('loading');
    btnText.textContent = 'Predicting…';
    result.classList.add('hidden');
    try{
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ email: text })
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
});
