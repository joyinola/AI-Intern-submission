/* ═══════════════════════════════════════════════════════════
   CHECK ME DESKTOP — Shared JS (model.js)
   State, presets, sidebar builder, and API client.

   The UI now calls the real FastAPI backend at /api/assess
   (proxied by Nginx from the same origin — no CORS issues).
   Falls back to the local JS model if the API is unreachable.
═══════════════════════════════════════════════════════════ */

/* ── API base URL ─────────────────────────────────────────
   When served through Docker / Nginx, the API is proxied
   at /api on the same host, so no hardcoded port is needed.
   For local file:// development, point at localhost:8000.  */
const API_BASE = (location.protocol === 'file:')
  ? 'http://localhost:8000'
  : '/api';

/* ── State via sessionStorage ── */
const DEFAULT_STATE = {
  age:45, bmi:26, alcohol_drinks_week:3,
  family_history_bc:0, brca_mutation:0, prior_biopsy:0,
  hrt_use:0, dense_breast:0, palpable_lump:0,
  nipple_discharge:0, skin_changes:0,
  radius_mean:13, texture_mean:19, perimeter_mean:87,
  area_mean:655, smoothness_mean:0.097, compactness_mean:0.10,
  concavity_mean:0.09, concave_points_mean:0.05,
  symmetry_mean:0.181, fractal_dim_mean:0.062,
  radius_worst:16.3, texture_worst:25.7, area_worst:880,
  concavity_worst:0.27, concave_pts_worst:0.11,
};

function saveState(s)  { sessionStorage.setItem('cm_state',  JSON.stringify(s)); }
function saveResult(r) { sessionStorage.setItem('cm_result', JSON.stringify(r)); }
function loadState()   { try { const s=sessionStorage.getItem('cm_state');  return s ? {...DEFAULT_STATE,...JSON.parse(s)} : {...DEFAULT_STATE}; } catch { return {...DEFAULT_STATE}; } }
function loadResult()  { try { const r=sessionStorage.getItem('cm_result'); return r ? JSON.parse(r) : null; } catch { return null; } }

/* ── Presets ── */
const PRESETS = {
  routine:  { age:42, bmi:24.5, alcohol_drinks_week:2,  family_history_bc:0, brca_mutation:0, prior_biopsy:0, hrt_use:0, dense_breast:0, palpable_lump:0, nipple_discharge:0, skin_changes:0, radius_mean:12.3, texture_mean:17.8, perimeter_mean:78.9, area_mean:476,  smoothness_mean:0.091, compactness_mean:0.076, concavity_mean:0.041, concave_points_mean:0.022, symmetry_mean:0.174, fractal_dim_mean:0.062, radius_worst:14.1, texture_worst:25.1, area_worst:572,  concavity_worst:0.12, concave_pts_worst:0.072 },
  elevated: { age:50, bmi:27.8, alcohol_drinks_week:6,  family_history_bc:1, brca_mutation:0, prior_biopsy:1, hrt_use:0, dense_breast:1, palpable_lump:0, nipple_discharge:0, skin_changes:0, radius_mean:14,   texture_mean:20.5, perimeter_mean:91.2, area_mean:620,  smoothness_mean:0.099, compactness_mean:0.098, concavity_mean:0.088, concave_points_mean:0.044, symmetry_mean:0.181, fractal_dim_mean:0.064, radius_worst:17.2, texture_worst:28.1, area_worst:840,  concavity_worst:0.28, concave_pts_worst:0.098 },
  high:     { age:58, bmi:30.1, alcohol_drinks_week:10, family_history_bc:1, brca_mutation:0, prior_biopsy:1, hrt_use:1, dense_breast:1, palpable_lump:1, nipple_discharge:0, skin_changes:0, radius_mean:17.2, texture_mean:22.1, perimeter_mean:114.8,area_mean:932,  smoothness_mean:0.108, compactness_mean:0.151, concavity_mean:0.168, concave_points_mean:0.091, symmetry_mean:0.194, fractal_dim_mean:0.064, radius_worst:21.4, texture_worst:30.1, area_worst:1438, concavity_worst:0.468,concave_pts_worst:0.187 },
  urgent:   { age:45, bmi:27,   alcohol_drinks_week:5,  family_history_bc:1, brca_mutation:1, prior_biopsy:0, hrt_use:0, dense_breast:1, palpable_lump:1, nipple_discharge:1, skin_changes:1, radius_mean:19.5, texture_mean:24,   perimeter_mean:128,  area_mean:1200, smoothness_mean:0.110, compactness_mean:0.180, concavity_mean:0.210, concave_points_mean:0.110, symmetry_mean:0.200, fractal_dim_mean:0.066, radius_worst:24,   texture_worst:32,   area_worst:2100, concavity_worst:0.75, concave_pts_worst:0.22  },
};

/* ── Tier display config ── */
const TIER_CFG = {
  ROUTINE:  { color:'#27ae60', lt:'#eafaf1', badge:'R', label:'Routine',  sub:'Standard Screening Pathway'    },
  ELEVATED: { color:'#f5a623', lt:'#fef9ec', badge:'E', label:'Elevated', sub:'Enhanced Monitoring Required'  },
  HIGH:     { color:'#c0392b', lt:'#fde8e7', badge:'H', label:'High',     sub:'Urgent Specialist Review'      },
  URGENT:   { color:'#8e44ad', lt:'#f5eef8', badge:'U', label:'Urgent',   sub:'Immediate Escalation Required' },
};

/* ── Local JS fallback (used if API is unreachable) ───────
   Keeps the same logic as the Python model for offline use. */
const CLINICAL_FLAGS = [
  { id:'brca_palpable', urgency:'IMMEDIATE', label:'Known BRCA mutation + palpable lump',  rationale:'ACS: immediate surgical referral recommended',           check:p=>p.brca_mutation&&p.palpable_lump },
  { id:'skin_nipple',   urgency:'IMMEDIATE', label:'Skin changes + nipple discharge',       rationale:'BI-RADS: associated with inflammatory breast cancer',    check:p=>p.skin_changes&&p.nipple_discharge },
  { id:'skin_change',   urgency:'IMMEDIATE', label:"Skin dimpling / peau d'orange",         rationale:'Classic clinical sign of underlying malignancy',        check:p=>p.skin_changes },
  { id:'brca_known',    urgency:'PROMPT',    label:'Known BRCA1/2 pathogenic variant',       rationale:'NICE NG101: high-risk surveillance pathway required',    check:p=>p.brca_mutation },
  { id:'fh_lump',       urgency:'PROMPT',    label:'Family history + palpable lump',         rationale:'ACS: combination warrants urgent imaging referral',      check:p=>p.family_history_bc&&p.palpable_lump },
  { id:'high_concavity',urgency:'PROMPT',    label:'Concavity worst > 0.70 (FNA)',           rationale:'High nuclear concavity — cytological malignancy marker', check:p=>p.concavity_worst>0.70 },
  { id:'large_area',    urgency:'PROMPT',    label:'Nuclear area worst > 2,000 µm²',        rationale:'Markedly enlarged nuclei — strong malignancy predictor', check:p=>p.area_worst>2000 },
  { id:'age_brca',      urgency:'PROMPT',    label:'Age > 30 with BRCA mutation',            rationale:'NICE NG101: annual MRI from age 30 for BRCA carriers',  check:p=>p.brca_mutation&&p.age>30 },
];

const WEIGHTS = {
  intercept:-5.8, concave_pts_worst:3.2, concavity_worst:2.4, area_worst:0.0018,
  radius_worst:0.22, concavity_mean:3.1, concave_points_mean:3.0, area_mean:0.0012,
  texture_worst:0.07, perimeter_mean:0.02, radius_mean:0.18, palpable_lump:1.4,
  brca_mutation:1.3, family_history_bc:0.9, skin_changes:1.1, nipple_discharge:0.85,
  dense_breast:0.55, prior_biopsy:0.45, hrt_use:0.50, age:0.022, bmi:0.025,
  alcohol_drinks_week:0.045, compactness_mean:1.2, smoothness_mean:4.5,
  symmetry_mean:0.8, fractal_dim_mean:-2.0, texture_mean:0.06,
};
const STATS = {
  radius_mean:{m:13,s:3}, texture_mean:{m:19,s:4}, perimeter_mean:{m:87,s:18},
  area_mean:{m:655,s:350}, smoothness_mean:{m:0.097,s:0.015}, compactness_mean:{m:0.10,s:0.05},
  concavity_mean:{m:0.09,s:0.08}, concave_points_mean:{m:0.05,s:0.04},
  symmetry_mean:{m:0.181,s:0.027}, fractal_dim_mean:{m:0.062,s:0.007},
  radius_worst:{m:16.3,s:4.8}, texture_worst:{m:25.7,s:6.2}, area_worst:{m:880,s:570},
  concavity_worst:{m:0.27,s:0.21}, concave_pts_worst:{m:0.11,s:0.07},
  age:{m:50,s:15}, bmi:{m:27,s:5.5}, alcohol_drinks_week:{m:3.5,s:4},
  family_history_bc:{m:0.18,s:0.38}, prior_biopsy:{m:0.20,s:0.40}, hrt_use:{m:0.24,s:0.43},
  brca_mutation:{m:0.07,s:0.25}, dense_breast:{m:0.37,s:0.48}, palpable_lump:{m:0.38,s:0.49},
  nipple_discharge:{m:0.14,s:0.35}, skin_changes:{m:0.10,s:0.30},
};
const TIER_RECS = {
  ROUTINE:  { recs:['Continue annual mammography screening per ACS age-appropriate guidelines','Monthly self-breast examination recommended','Maintain healthy weight and limit alcohol to < 7 units/week','Next scheduled mammogram as per standard programme'], steps:['Routine screening mammogram (annual if ≥ 40)','Clinical breast exam at next GP visit','Reassess risk score if new symptoms develop'] },
  ELEVATED: { recs:['Earlier or supplemental screening discussion with GP/oncologist','Consider breast ultrasound as adjunct to mammography','Document and monitor all risk factors at follow-up visits','Discuss lifestyle modifications: weight management, alcohol reduction'], steps:['GP referral for clinical breast assessment within 6 weeks','Discuss supplemental MRI if dense breast tissue confirmed','Genetic counselling referral if family history suggests hereditary risk'] },
  HIGH:     { recs:['Prompt specialist referral to breast clinic or oncology unit','Annual breast MRI alongside mammography (NICE NG101)','Discussion of risk-reduction options (chemoprevention, lifestyle)','Formal genetic risk assessment recommended'], steps:['Urgent breast clinic referral within 2 weeks','Diagnostic mammogram + ultrasound evaluation','Genetic counselling if BRCA/family risk present','Oncology nurse specialist involvement'] },
  URGENT:   { recs:['⚠ Immediate specialist referral — do not delay','Two-week-wait (2WW) urgent cancer pathway referral','Physical exam findings require prompt imaging workup','Do not reassure without complete imaging and clinical evaluation'], steps:['2WW urgent breast clinic referral TODAY','Triple assessment: clinical exam + mammogram + FNA/core biopsy','MDT review of imaging and pathology','Patient counselling and psychological support resources'] },
};

function sigmoid(x) { return 1/(1+Math.exp(-x)); }

function _localScore(p) {
  let logit = WEIGHTS.intercept;
  for (const [f,w] of Object.entries(WEIGHTS)) {
    if (f==='intercept') continue;
    const st = STATS[f]||{m:0,s:1};
    logit += w*(((p[f]??0)-st.m)/(st.s||1));
  }
  const prob = Math.max(0.01,Math.min(0.99,sigmoid(logit)));
  const flags = CLINICAL_FLAGS.filter(f=>f.check(p));
  const hasImm = flags.some(f=>f.urgency==='IMMEDIATE');
  const hasPrm = flags.some(f=>f.urgency==='PROMPT');
  const tier = hasImm||prob>=0.72 ? 'URGENT' : hasPrm||prob>=0.48 ? 'HIGH' : prob>=0.28 ? 'ELEVATED' : 'ROUTINE';
  const d = Math.abs(prob-0.5);
  const confidence = (d>0.30||hasImm)?'HIGH':d>0.12?'MODERATE':'LOW';
  const recs = TIER_RECS[tier];
  return {
    tier, risk_score: prob, confidence,
    clinical_flags: flags.map(f=>({id:f.id,condition:f.label,rationale:f.rationale,urgency:f.urgency})),
    top_features: [], recommendations: recs.recs, next_steps: recs.steps,
    disclaimer: 'Check Me is a clinical decision SUPPORT tool. All outputs require validation by a qualified clinician.',
    source: 'local_fallback',
  };
}

/* ── Main API call ─────────────────────────────────────────
   Returns a normalised result object whether from the API
   or from the local JS fallback.                          */
async function scorePatient(p) {
  try {
    const resp = await fetch(`${API_BASE}/assess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
      signal: AbortSignal.timeout(8000),
    });
    if (!resp.ok) throw new Error(`API ${resp.status}`);
    const data = await resp.json();
    // Normalise API response to the shape the UI expects
    return {
      tier: data.risk_tier,
      risk_score: data.risk_score,
      confidence: data.confidence,
      clinical_flags: data.clinical_flags,
      top_features: data.top_features || [],
      recommendations: data.recommendations,
      next_steps: data.next_steps,
      disclaimer: data.disclaimer,
      source: 'api',
    };
  } catch (err) {
    console.warn('Check Me API unreachable — using local fallback model.', err.message);
    return _localScore(p);
  }
}

/* ── Sidebar builder ── */
const NAV_ITEMS = [
  { id:'dashboard', label:'Dashboard',     href:'index.html',      svg:'<path d="M3 4a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 12a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H4a1 1 0 01-1-1v-4zM13 3a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V4a1 1 0 00-1-1h-4zM12 12a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"/>' },
  { id:'assess',    label:'New Assessment',href:'assess.html',     svg:'<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>' },
  { id:'results',   label:'Results',       href:'results.html',    svg:'<path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/>' },
  { id:'flags',     label:'Safety Flags',  href:'flags.html',      svg:'<path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>' },
  { id:'guidelines',label:'Guidelines',    href:'guidelines.html', svg:'<path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/>' },
];

function buildSidebar(active) {
  const links = NAV_ITEMS.map(n=>`
    <a class="nav-link${n.id===active?' active':''}" href="${n.href}">
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
