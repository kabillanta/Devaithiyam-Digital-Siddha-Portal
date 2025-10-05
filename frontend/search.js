document.addEventListener('DOMContentLoaded', () => {
  const diseaseInput = document.getElementById('disease-input');
  const ingredientsInput = document.getElementById('ingredients-input');
  const runBtn = document.getElementById('run-search');
  const clearBtn = document.getElementById('clear-search');

  const diseaseResults = document.getElementById('disease-results');
  const diseasesForIngredientsEl = document.getElementById('diseases-for-ingredients');
  const ingredientRemedies = document.getElementById('ingredient-remedies');
  // column wrappers for conditional display
  const colDisease = document.getElementById('col-disease');
  const colDisIng = document.getElementById('col-diseases-for-ingredients');
  const colIngRem = document.getElementById('col-ingredient-remedies');

  function escapeHtml(str) {
    return (str || '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
  }
  function renderList(el, rows, type, matchInfo) {
    el.innerHTML = '';
    if (!rows || rows.length === 0) {
      el.innerHTML = '<li class="muted">No results</li>';
      return;
    }
    const frag = document.createDocumentFragment();
    rows.forEach((row, idx) => {
  const li = document.createElement('li');
      if (type === 'disease') {
        const name = escapeHtml(row.disease || row.name || 'Unknown');
        const symptoms = escapeHtml(row.sign_and_symptoms || '');
        const score = row.score !== undefined ? `<br/><small>Score: ${escapeHtml(String(row.score))}${row.token_coverage !== undefined ? ' | Coverage: ' + escapeHtml(String(row.token_coverage)) : ''}${row.matched_ingredients ? ' | Ingredients: ' + escapeHtml(row.matched_ingredients.join(', ')) : ''}</small>` : '';
        li.innerHTML = `<strong>${name}</strong><br/><small>${symptoms}</small>${score}`;
      } else if (type === 'remedy') {
        const name = escapeHtml(row['Remedy Name'] || 'Remedy');
        const prep = escapeHtml(row['Preparation'] || '');
        const usage = escapeHtml(row['Usage'] || '');
        const url = `remedies.html?query=${encodeURIComponent(name)}`;
        let matchLine = '';
        if (matchInfo && matchInfo[idx]) {
          const mi = matchInfo[idx];
          const matched = (mi.matched || []).join(', ');
          const missing = (mi.missing || []).join(', ');
          matchLine = `<br/><small>Matched: ${escapeHtml(matched)}${missing ? ' | Missing: ' + escapeHtml(missing) : ''}${mi.coverage !== undefined ? ' | Coverage: ' + escapeHtml(String(mi.coverage)) : ''}${mi.score !== undefined ? ' | Score: ' + escapeHtml(String(mi.score)) : ''}</small>`;
        }
        const scoreBadge = row.score !== undefined ? `<br/><small>Overall Score: ${escapeHtml(String(row.score))}</small>` : '';
        li.innerHTML = `<strong>${name}</strong><br/><small>Prep: ${prep}</small><br/><small>Usage: ${usage}</small>${matchLine}${scoreBadge}`;
      }
      frag.appendChild(li);
    });
    el.appendChild(frag);
  }

  async function runSearch() {
    const disease = diseaseInput.value.trim();
    const ingredients = ingredientsInput.value.trim();
    const url = new URL('http://127.0.0.1:8000/search/filters');
    if (disease) url.searchParams.set('disease', disease);
    if (ingredients) url.searchParams.set('ingredients', ingredients);

    runBtn.disabled = true;
    runBtn.textContent = 'Searching...';

    try {
      const res = await fetch(url.toString());
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      // Determine which sections to show
      const hasDiseaseQuery = !!disease;
      const hasIngredientQuery = !!ingredients;

      // Clear & hide all first
  [colDisease, colDisIng, colIngRem].forEach(c => { if (c) c.style.display = 'none'; });
      diseaseResults.innerHTML = '';
      diseasesForIngredientsEl.innerHTML = '';
      ingredientRemedies.innerHTML = '';

      if (hasDiseaseQuery) {
        colDisease.style.display = 'block';
        renderList(diseaseResults, data.disease_matches, 'disease');
        const singlePrimary = data.disease_matches && data.disease_matches.length === 1 && data.disease_matches[0].primary;
        if (singlePrimary) {
          // Render embedded remedies under the disease list item
          const primaryLi = diseaseResults.querySelector('li');
          if (primaryLi) {
            const remedies = data.disease_matches[0].remedies || [];
            if (remedies.length) {
              const sub = document.createElement('ul');
              sub.className = 'nested-remedies';
              remedies.forEach(r => {
                const ri = document.createElement('li');
                ri.classList.add('result-card-item');
                ri.innerHTML = `<strong>${escapeHtml(r.name || 'Remedy')}</strong><br/><small>Preparation: ${escapeHtml(r.preparation || '')}</small><br/><small>Usage: ${escapeHtml(r.usage || '')}</small>`;
                sub.appendChild(ri);
              });
              primaryLi.appendChild(sub);
            }
          }
        }
      }
      if (hasIngredientQuery) {
        colIngRem.style.display = 'block';
        renderList(ingredientRemedies, data.remedies_using_ingredients, 'remedy', data.ingredient_match_info);
        if (!hasDiseaseQuery) {
          colDisIng.style.display = 'block';
          renderList(diseasesForIngredientsEl, data.diseases_for_ingredients, 'disease');
        }
      }
      // If neither (shouldn't happen because user can click with empty) show message
      if (!hasDiseaseQuery && !hasIngredientQuery) {
        colDisease.style.display = 'block';
        diseaseResults.innerHTML = '<li class="muted">Enter a disease, ingredients, or both.</li>';
      }
    } catch (e) {
      console.error(e);
  [colDisease, colDisIng, colIngRem].forEach(c => { if (c) c.style.display = 'block'; });
      diseaseResults.innerHTML = '<li class="error">Search failed. Check server.</li>';
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = 'Search';
    }
  }

  runBtn.addEventListener('click', runSearch);
  clearBtn.addEventListener('click', () => {
    diseaseInput.value = '';
    ingredientsInput.value = '';
  [colDisease, colDisIng, colIngRem].forEach(c => { if (c) c.style.display = 'none'; });
  });
});
