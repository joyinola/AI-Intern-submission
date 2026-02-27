/* ═══════════════════════════════════════════════════════════════════════
   CHECK ME — Shared JS (model.js)
   State management, presets, API client, local fallback, sidebar builder.

   Model:    GradientBoostingClassifier (sklearn, XGBoost-equivalent)
   Dataset:  Real UCI Wisconsin BC Diagnostic — 569 patients
   Bands:    GREEN | YELLOW | RED
   API:      POST /api/assess  (FastAPI via Nginx proxy)
             Falls back to local JS approximation if API is unreachable.
═══════════════════════════════════════════════════════════════════════ */

const API_BASE = (location.protocol === 'file:')
  ? 'http://localhost:8000'
  : '/api';

/* ─── Default form state ────────────────────────────────────────────── */
const DEFAULT_STATE = {
  age: 45, bmi: 26, alcohol_drinks_week: 3,
  family_history_bc: 0, brca_mutation: 0, prior_biopsy: 0,
  hrt_use: 0, dense_breast: 0, palpable_lump: 0,
  nipple_discharge: 0, skin_changes: 0,
  mean_radius: 14.1, mean_texture: 19.3, mean_perimeter: 91.97,
  mean_area: 654.9, mean_smoothness: 0.0964, mean_compactness: 0.1043,
  mean_concavity: 0.0888, mean_concave_points: 0.0489,
  mean_symmetry: 0.181, mean_fractal_dimension: 0.0628,
  worst_radius: 16.3, worst_texture: 25.7, worst_area: 880.6,
  worst_concavity: 0.2722, worst_concave_points: 0.1147,
};

function saveState(s)  { sessionStorage.setItem('cm_state',  JSON.stringify(s)); }
function saveResult(r) { sessionStorage.setItem('cm_result', JSON.stringify(r)); }
function loadState()   { try { const s = sessionStorage.getItem('cm_state');  return s ? {...DEFAULT_STATE, ...JSON.parse(s)} : {...DEFAULT_STATE}; } catch { return {...DEFAULT_STATE}; } }
function loadResult()  { try { const r = sessionStorage.getItem('cm_result'); return r ? JSON.parse(r) : null; } catch { return null; } }

/* ─── Example presets (GREEN / YELLOW / RED) ───────────────────────── */
const PRESETS = {
  green: {
    age: 34, bmi: 22.8, alcohol_drinks_week: 1,
    family_history_bc: 0, brca_mutation: 0, prior_biopsy: 0, hrt_use: 0,
    dense_breast: 0, palpable_lump: 0, nipple_discharge: 0, skin_changes: 0,
    mean_radius: 10.9, mean_texture: 15.7, mean_perimeter: 70.3,
    mean_area: 374.0, mean_smoothness: 0.083, mean_compactness: 0.058,
    mean_concavity: 0.022, mean_concave_points: 0.015,
    mean_symmetry: 0.168, mean_fractal_dimension: 0.059,
    worst_radius: 12.4, worst_texture: 21.6, worst_area: 477.0,
    worst_concavity: 0.068, worst_concave_points: 0.042,
  },
  yellow: {
    // Benign FNA values + meaningful clinical risk factors → YELLOW via crs
    age: 52, bmi: 27.0, alcohol_drinks_week: 5,
    family_history_bc: 1, brca_mutation: 0, prior_biopsy: 1, hrt_use: 1,
    dense_breast: 1, palpable_lump: 0, nipple_discharge: 0, skin_changes: 0,
    mean_radius: 13.5, mean_texture: 18.9, mean_perimeter: 87.2,
    mean_area: 567.0, mean_smoothness: 0.091, mean_compactness: 0.083,
    mean_concavity: 0.055, mean_concave_points: 0.029,
    mean_symmetry: 0.176, mean_fractal_dimension: 0.061,
    worst_radius: 15.3, worst_texture: 24.8, worst_area: 726.0,
    worst_concavity: 0.182, worst_concave_points: 0.074,
  },
  red: {
    age: 58, bmi: 30.4, alcohol_drinks_week: 10,
    family_history_bc: 1, brca_mutation: 1, prior_biopsy: 1, hrt_use: 1,
    dense_breast: 1, palpable_lump: 1, nipple_discharge: 1, skin_changes: 1,
    mean_radius: 19.8, mean_texture: 24.9, mean_perimeter: 131.2,
    mean_area: 1228.0, mean_smoothness: 0.113, mean_compactness: 0.188,
    mean_concavity: 0.224, mean_concave_points: 0.119,
    mean_symmetry: 0.202, mean_fractal_dimension: 0.067,
    worst_radius: 25.1, worst_texture: 33.4, worst_area: 2486.0,
    worst_concavity: 0.786, worst_concave_points: 0.238,
  },
};

/* ─── Band display config ───────────────────────────────────────────── */
const TIER_CFG = {
  GREEN:  { color: '#27ae60', lt: '#eafaf1', badge: 'G', label: 'Green',  sub: 'Low Concern — Keep Up the Good Work'       },
  YELLOW: { color: '#f5a623', lt: '#fef9ec', badge: 'Y', label: 'Yellow', sub: 'Some Risk Factors Present — Worth a Check-Up'  },
  RED:    { color: '#c0392b', lt: '#fde8e7', badge: 'R', label: 'Red',    sub: 'Please See a Doctor Soon'       },
};

/* ─── Band recommendations (mirrors decision.py exactly) ───────────── */
const TIER_RECS = {
  GREEN: {
    recs: [
      'Your results look reassuring. Keep up your regular breast health habits.',
      'Continue routine mammography screening as recommended for your age group.',
      'Do a monthly self-check so you know what feels normal for you.',
      'Maintain a balanced lifestyle — healthy weight, limited alcohol, regular movement.',
    ],
    steps: [
      'Schedule your next routine mammogram as usual',
      'Mention this assessment at your next GP check-up',
      'Come back to reassess if you notice any new changes',
    ],
  },
  YELLOW: {
    recs: [
      'Some risk factors in your profile are worth discussing with your doctor.',
      'This is not a cause for alarm — many people with similar profiles are healthy.',
      'A check-up will give you clarity and peace of mind.',
      'Small lifestyle adjustments (reducing alcohol, maintaining a healthy weight) can make a meaningful difference.',
    ],
    steps: [
      'Book a GP appointment in the next few weeks',
      'Mention your family history and any risk factors at the appointment',
      'Ask your doctor whether additional screening is right for you',
    ],
  },
  RED: {
    recs: [
      'Some of your results need to be checked by a healthcare professional.',
      'This is not a diagnosis — only a doctor can properly evaluate your health.',
      'Getting checked early is always the right move, whatever the outcome.',
      'You don\'t have to face this alone — bring someone you trust to your appointment.',
    ],
    steps: [
      'Contact your GP or local breast clinic as soon as you can',
      'Describe your symptoms and mention this assessment',
      'Ask about a breast examination and any imaging they recommend',
    ],
  },
};

/* ─── Global permutation importance (real UCI model) ───────────────── */
const GLOBAL_IMP = {
  worst_area:           0.00752,
  worst_concave_points: 0.000826,
  mean_concave_points:  0.000119,
  worst_texture:        0.0000793,
  mean_compactness:     0.0000529,
  mean_radius:          0,
  mean_texture:         0,
  mean_perimeter:       0,
  mean_area:            0,
  mean_smoothness:      0,
  mean_concavity:       0,
  mean_symmetry:        0,
  worst_radius:         0,
  worst_concavity:      0,
};

/* ─── Clinical safety flags (mirrors flags.py) ─────────────────────── */
const CLINICAL_FLAGS = [
  { id: 'brca_palpable', urgency: 'IMMEDIATE', label: 'Known BRCA mutation + palpable lump',  rationale: 'ACS: immediate surgical referral recommended',           check: p => p.brca_mutation && p.palpable_lump },
  { id: 'skin_nipple',   urgency: 'IMMEDIATE', label: 'Skin changes + nipple discharge',       rationale: 'BI-RADS: combined signs associated with inflammatory BC', check: p => p.skin_changes && p.nipple_discharge },
  { id: 'skin_change',   urgency: 'IMMEDIATE', label: "Skin dimpling / peau d'orange",         rationale: 'Classic clinical sign of underlying malignancy',         check: p => p.skin_changes },
  { id: 'brca_known',    urgency: 'PROMPT',    label: 'Known BRCA1/2 pathogenic variant',      rationale: 'NICE NG101: high-risk surveillance pathway required',     check: p => p.brca_mutation },
  { id: 'fh_lump',       urgency: 'PROMPT',    label: 'Family history + palpable lump',        rationale: 'ACS: combination warrants urgent imaging referral',       check: p => p.family_history_bc && p.palpable_lump },
  { id: 'high_concavity',urgency: 'PROMPT',    label: 'Worst concavity > 0.70 (FNA)',          rationale: 'High nuclear concavity — cytological malignancy marker',  check: p => (p.worst_concavity || 0) > 0.70 },
  { id: 'large_area',    urgency: 'PROMPT',    label: 'Worst nuclear area > 2,000 µm²',        rationale: 'Markedly enlarged nuclei — strong malignancy predictor',  check: p => (p.worst_area || 0) > 2000 },
  { id: 'age_brca',      urgency: 'PROMPT',    label: 'Age > 30 with BRCA mutation',           rationale: 'NICE: annual MRI from age 30 for BRCA carriers',          check: p => p.brca_mutation && (p.age || 0) > 30 },
];

/* ─── Clinical risk score (mirrors decision.py) ────────────────────── */
function _clinicalRiskScore(p) {
  let s = 0;
  if (p.palpable_lump)                s += 3;
  if (p.skin_changes)                 s += 3;
  if (p.nipple_discharge)             s += 2;
  if (p.brca_mutation)                s += 3;
  if (p.family_history_bc)            s += 2;
  if (p.dense_breast)                 s += 1;
  if (p.hrt_use)                      s += 1;
  if (p.prior_biopsy)                 s += 1;
  if ((p.age || 0) >= 50)             s += 1;
  if ((p.age || 0) >= 60)             s += 1;
  if ((p.alcohol_drinks_week||0) >= 7) s += 1;
  return s;
}

/* ─── Local fallback (logistic approximation of the GB model) ──────── */
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

const _W = {
  // Logistic regression fitted to GB model outputs (AUC 0.998 approximation).
  // Weights are on z-scored features using _S means/stds below.
  // DO NOT hand-edit — regenerate from train_and_export.py if model changes.
  intercept: -0.8466,
  mean_radius: 0.3894,
  mean_texture: 0.4571,
  mean_perimeter: 0.3132,
  mean_area: 0.5502,
  mean_smoothness: 1.2147,
  mean_compactness: -0.9852,
  mean_concavity: 0.1392,
  mean_concave_points: 0.9864,
  mean_symmetry: 0.3804,
  mean_fractal_dimension: -0.4186,
  worst_radius: 1.6337,
  worst_texture: 1.0107,
  worst_area: 1.4505,
  worst_concavity: 1.1684,
  worst_concave_points: 1.1341,
  age: 0.6637,
  bmi: 0.5245,
  alcohol_drinks_week: -0.4066,
  family_history_bc: 0.0358,
  prior_biopsy: 0.4148,
  hrt_use: 0.2671,
  brca_mutation: 0.1161,
  dense_breast: 0.4375,
  palpable_lump: 0.9639,
  nipple_discharge: 0.5812,
  skin_changes: 0.0814,
};
const _S = {
  // Population means and stds from real UCI dataset (n=569).
  // Used to z-score inputs before applying _W weights.
  mean_radius:            { m: 14.127292, s: 3.520951  },
  mean_texture:           { m: 19.289649, s: 4.297255  },
  mean_perimeter:         { m: 91.969033, s: 24.277619 },
  mean_area:              { m: 654.889104,s: 351.604754 },
  mean_smoothness:        { m: 0.09636,   s: 0.014052  },
  mean_compactness:       { m: 0.104341,  s: 0.052766  },
  mean_concavity:         { m: 0.088799,  s: 0.07965   },
  mean_concave_points:    { m: 0.048919,  s: 0.038769  },
  mean_symmetry:          { m: 0.181162,  s: 0.02739   },
  mean_fractal_dimension: { m: 0.062798,  s: 0.007054  },
  worst_radius:           { m: 16.26919,  s: 4.828993  },
  worst_texture:          { m: 25.677223, s: 6.140854  },
  worst_area:             { m: 880.583128,s: 568.856459 },
  worst_concavity:        { m: 0.272188,  s: 0.208441  },
  worst_concave_points:   { m: 0.114606,  s: 0.065675  },
  age:                    { m: 50.286819, s: 14.316613 },
  bmi:                    { m: 26.973989, s: 5.504677  },
  alcohol_drinks_week:    { m: 3.039895,  s: 2.909383  },
  family_history_bc:      { m: 0.198594,  s: 0.398942  },
  prior_biopsy:           { m: 0.202109,  s: 0.401573  },
  hrt_use:                { m: 0.149385,  s: 0.356467  },
  brca_mutation:          { m: 0.066784,  s: 0.249647  },
  dense_breast:           { m: 0.370826,  s: 0.483026  },
  palpable_lump:          { m: 0.405975,  s: 0.49108   },
  nipple_discharge:       { m: 0.133568,  s: 0.340187  },
  skin_changes:           { m: 0.119508,  s: 0.324385  },
};

function _localScore(p) {
  let logit = _W.intercept;
  for (const [f, w] of Object.entries(_W)) {
    if (f === 'intercept') continue;
    const st = _S[f] || { m: 0, s: 1 };
    logit += w * (((p[f] ?? 0) - st.m) / (st.s || 1));
  }
  const prob = Math.max(0.01, Math.min(0.99, sigmoid(logit)));
  const flags    = CLINICAL_FLAGS.filter(f => f.check(p));
  const hasImm   = flags.some(f => f.urgency === 'IMMEDIATE');
  const hasPrm   = flags.some(f => f.urgency === 'PROMPT');
  const crs      = _clinicalRiskScore(p);
  let tier;
  if (hasImm || prob >= 0.85)                       tier = 'RED';
  else if ((prob >= 0.50 && crs >= 3) || hasPrm)   tier = 'RED';
  else if (prob >= 0.50 || crs >= 3)                tier = 'YELLOW';
  else                                              tier = 'GREEN';
  const confidence = (hasImm || prob >= 0.85) ? 'HIGH' : (prob >= 0.50 || crs >= 6) ? 'MODERATE' : 'LOW';
  const recs     = TIER_RECS[tier];
  return {
    tier, risk_score: prob, confidence,
    clinical_flags: flags.map(f => ({ id: f.id, condition: f.label, rationale: f.rationale, urgency: f.urgency })),
    top_features: [],
    recommendations: recs.recs, next_steps: recs.steps,
    disclaimer: 'This is not a diagnosis. Check Me is a self-screening support tool only. It cannot tell you whether you have or don\'t have cancer. For any concerns, please consult a healthcare professional.',
    source: 'local_fallback',
  };
}

/* ─── Main API call with local fallback ────────────────────────────── */
async function scorePatient(p) {
  try {
    const resp = await fetch(`${API_BASE}/assess`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(p),
      signal:  AbortSignal.timeout(8000),
    });
    if (!resp.ok) throw new Error(`API ${resp.status}`);
    const data = await resp.json();
    return {
      tier:          data.risk_tier,
      risk_score:    data.risk_score,
      confidence:    data.confidence,
      clinical_flags: data.clinical_flags,
      top_features:  data.top_features || [],
      recommendations: data.recommendations,
      next_steps:    data.next_steps,
      disclaimer:    data.disclaimer,
      source:        'api',
    };
  } catch (err) {
    console.warn('Check Me API unreachable — using local fallback.', err.message);
    return _localScore(p);
  }
}

/* ─── Sidebar ───────────────────────────────────────────────────────── */
const NAV_ITEMS = [
  { id: 'dashboard',  label: 'Dashboard',      href: 'index.html',      svg: '<path d="M3 4a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 12a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H4a1 1 0 01-1-1v-4zM13 3a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V4a1 1 0 00-1-1h-4zM12 12a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"/>' },
  { id: 'assess',     label: 'New Assessment', href: 'assess.html',     svg: '<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>' },
  { id: 'results',    label: 'Results',         href: 'results.html',    svg: '<path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/>' },
  { id: 'flags',      label: 'Safety Flags',    href: 'flags.html',      svg: '<path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>' },
  { id: 'guidelines', label: 'Guidelines',      href: 'guidelines.html', svg: '<path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/>' },
];

function buildSidebar(active) {
  const links = NAV_ITEMS.map(n => `
    <a class="nav-link${n.id === active ? ' active' : ''}" href="${n.href}">
      <span class="nav-ico"><svg viewBox="0 0 20 20" fill="currentColor">${n.svg}</svg></span>
      <span class="nav-lbl">${n.label}</span>
    </a>`).join('');
  return `
  <aside class="sidebar">
    <div class="sidebar-brand">
      <div class="brand-row">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24"><path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z"/></svg>
        </div>
        <div class="brand-name">Check<em>Me</em></div>
      </div>
      <div class="brand-tag">Breast Cancer Risk AI</div>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-section">Navigation</div>
      ${links}
    </nav>
    <div class="sidebar-foot">
      <div class="demo-chip">⚕ Research Demo Only</div>
    </div>
  </aside>`;
}
