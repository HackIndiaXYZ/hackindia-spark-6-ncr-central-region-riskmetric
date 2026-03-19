// ===== controllers/intelController.js =====
// Orchestrates: API gathering → Python data pipeline → ML scoring → response

const { gatherIntelligence } = require('../services/intelligenceService');
const { spawn }              = require('child_process');
const path                   = require('path');
const config                 = require('../config/config');

const ML_DIR = path.join(__dirname, '..', '..', 'ml');

// ── Helper: call a Python script via stdin/stdout JSON bridge ───────────────
function runPython(script, inputObj) {
  return new Promise((resolve, reject) => {
    const pythonCmd = config.PYTHON_CMD;
    const scriptPath = path.join(ML_DIR, script);

    const proc = spawn(pythonCmd, [scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', d => { stdout += d.toString(); });
    proc.stderr.on('data', d => { stderr += d.toString(); });

    proc.on('close', code => {
      if (code !== 0) {
        // Only reject on non-zero exit — stderr alone is not fatal
        return reject(new Error(`Python (${script}) exited ${code}: ${stderr.slice(0, 300)}`));
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        reject(new Error(`Python (${script}) returned invalid JSON: ${stdout.slice(0, 200)}`));
      }
    });

    proc.on('error', err => reject(new Error(`Could not spawn Python: ${err.message}`)));

    proc.stdin.write(JSON.stringify(inputObj));
    proc.stdin.end();
  });
}

// ── GET /api/health ──────────────────────────────────────────────────────────
async function health(req, res) {
  res.json({
    status: 'ok',
    version: '3.0.0',
    leakhunterKeySet: Boolean(config.LEAKHUNTER_API_KEY),
    pythonCmd: config.PYTHON_CMD,
    dataMode: config.DATA_MODE,
    timestamp: new Date().toISOString(),
  });
}

// ── POST /api/analyze ────────────────────────────────────────────────────────
async function getIntel(req, res) {
  const { identifier } = req.body;

  try {
    // 1. Gather from both APIs in parallel
    const intel = await gatherIntelligence(identifier);
    const { features, leakhunter, xposedornot, merged } = intel;

    // 2. Run data_pipeline.py (log1p normalisation + clamping)
    let pipelineOut;
    try {
      pipelineOut = await runPython('data_pipeline.py', { features });
    } catch (pyErr) {
      console.warn('[Pipeline] Python unavailable — using heuristic fallback:', pyErr.message);
      pipelineOut = heuristicPipeline(features);
    }

    // 3. Run model.py (RandomForest + IsolationForest + SHAP)
    let mlOut;
    try {
      mlOut = await runPython('model.py', { normalized_features: pipelineOut });
    } catch (pyErr) {
      console.warn('[Model] Python unavailable — using heuristic fallback:', pyErr.message);
      mlOut = heuristicModel(pipelineOut, merged.signals);
    }

    // 4. Build response
    const response = {
      identifier,
      leakhunter,
      xposedornot,
      ml: {
        finalScore:  mlOut.score,
        riskLevel:   mlOut.risk_level,
        factors:     mlOut.factors,
        shapFactors: mlOut.shap_factors,
        meta: {
          breachCount:   merged.breachCount,
          passwordLeaks: merged.passwordLeaks,
          avgSeverity:   parseFloat(features.avg_severity.toFixed(1)),
          criticalCount: features.critical_count,
          recentBreaches: features.recent_breaches,
        },
      },
      quota: leakhunter?.quota || null,
      apiCoverage: {
        leakhunter:  leakhunter ? 'ok' : 'error',
        xposedornot: xposedornot ? 'ok' : 'error',
      },
      timestamp: new Date().toISOString(),
    };

    return res.json(response);

  } catch (err) {
    console.error('[IntelController] Fatal error:', err);
    return res.status(500).json({ error: err.message, detail: err.stack?.split('\n')[0] });
  }
}

// ── Heuristic fallback if Python is unavailable ──────────────────────────────
function heuristicPipeline(features) {
  const log1p = x => Math.log(1 + x);
  return {
    breach_norm:       Math.min(log1p(features.breach_count) / log1p(200), 1),
    password_norm:     Math.min(log1p(features.password_leaks) / log1p(100), 1),
    severity_norm:     Math.min(features.avg_severity / 10, 1),
    critical_norm:     Math.min(log1p(features.critical_count) / log1p(50), 1),
    recent_norm:       Math.min(log1p(features.recent_breaches) / log1p(30), 1),
    login_anomaly_score:   Math.min(Math.max(features.login_anomaly_score, 0), 1),
    public_exposure:       Math.min(Math.max(features.public_exposure, 0), 1),
    social_risk_score:     Math.min(Math.max(features.social_risk_score, 0), 1),
    has_password_breach:   features.has_password_breach,
  };
}

function heuristicModel(nf, signals) {
  const dangerComposite = nf.breach_norm * 0.3 + nf.password_norm * 0.25 + nf.critical_norm * 0.25 + nf.severity_norm * 0.2;
  const pHigh = Math.min(dangerComposite * 1.1, 1);
  const pLow  = Math.max(1 - pHigh - 0.15, 0);
  const pMed  = 1 - pHigh - pLow;
  let score   = pLow * 18 + pMed * 52 + pHigh * 86;
  if (signals?.highSeverityBreach) score += 6;
  score = Math.min(Math.round(score), 100);

  const risk = score >= 75 ? 'CRITICAL' : score >= 50 ? 'HIGH RISK' : score >= 25 ? 'MEDIUM' : 'LOW RISK';

  return {
    score,
    risk_level: risk,
    factors: [
      { icon: '🔑', name: 'PASSWORD LEAKS',  score: Math.round(nf.password_norm * 10), barColor: 'var(--accent-red)' },
      { icon: '💀', name: 'BREACH SEVERITY', score: Math.round(nf.severity_norm * 10), barColor: 'var(--accent-orange)' },
      { icon: '🔁', name: 'EXPOSURE COUNT',  score: Math.round(nf.breach_norm * 10),   barColor: 'var(--accent-yellow)' },
      { icon: '⚡', name: 'RECENT BREACHES', score: Math.round(nf.recent_norm * 10),   barColor: 'var(--accent-cyan)' },
      { icon: '🌐', name: 'PUBLIC EXPOSURE', score: Math.round(nf.public_exposure * 10), barColor: 'var(--accent-blue)' },
    ],
    shap_factors: [
      { label: 'has_password_breach',  pts: Math.round(nf.has_password_breach * 28 + nf.password_norm * 10), pct: 80 },
      { label: 'breach_severity_score', pts: Math.round(nf.severity_norm * 22 + nf.critical_norm * 8), pct: 65 },
      { label: 'breach_norm (log1p)',  pts: Math.round(nf.breach_norm * 18), pct: 55 },
      { label: 'login_anomaly_score',  pts: Math.round(nf.login_anomaly_score * 15), pct: 40 },
      { label: 'danger_composite',     pts: Math.round(dangerComposite * 12), pct: 35 },
      { label: 'public_exposure',      pts: Math.round(nf.public_exposure * 8), pct: 25 },
      { label: 'social_risk_score',    pts: Math.round(nf.social_risk_score * 6), pct: 20 },
      { label: 'recent_breach_norm',   pts: Math.round(nf.recent_norm * 10), pct: 30 },
      { label: 'password_norm (log1p)',pts: Math.round(nf.password_norm * 14), pct: 45 },
    ].sort((a,b) => b.pts - a.pts),
  };
}

module.exports = { getIntel, health };
