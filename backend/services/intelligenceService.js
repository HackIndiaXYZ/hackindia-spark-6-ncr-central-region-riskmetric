// ===== services/intelligenceService.js =====
// Merges LeakHunter AI + XposedOrNot results
// Derives the 9-feature vector for the ML pipeline

const { checkLeakHunter } = require('./leakhunterService');
const { checkXON }         = require('./xonService');

async function gatherIntelligence(email) {
  // Run both APIs in parallel for maximum coverage
  const [lhResult, xonResult] = await Promise.allSettled([
    checkLeakHunter(email),
    checkXON(email),
  ]);

  const lh  = lhResult.status  === 'fulfilled' ? lhResult.value  : null;
  const xon = xonResult.status === 'fulfilled' ? xonResult.value : null;

  // ── Merge ──────────────────────────────────────────────────────────────────
  const exposures     = lh?.exposures || [];
  const signals       = lh?.signals   || {};
  const lhBreachCount = lh?.exposureCount || exposures.length || 0;
  const xonBreachCount = xon?.breachCount || 0;
  const mergedBreachCount = Math.max(lhBreachCount, xonBreachCount);

  const passwordLeaks = exposures.filter(e => e.passwordIncluded).length
    || Math.max(xon?.passwordLeakCount || 0, 0);

  const avgSeverity = exposures.length > 0
    ? exposures.reduce((s, e) => s + e.severity, 0) / exposures.length
    : 0;

  const criticalCount = exposures.filter(e => e.severity >= 9).length;

  const recentBreaches = exposures.filter(e => {
    const y = parseInt((e.date || '0000').substring(0, 4));
    return y >= 2020;
  }).length;

  // ── 9 ML Features (raw, before normalisation) ───────────────────────────
  const features = {
    breach_count:          mergedBreachCount,
    password_leaks:        passwordLeaks,
    avg_severity:          avgSeverity,
    critical_count:        criticalCount,
    recent_breaches:       recentBreaches,
    login_anomaly_score:   signals.recentBreach ? 0.75 : 0.2,
    public_exposure:       Math.min(mergedBreachCount / 200, 1),
    social_risk_score:     signals.multipleExposures ? 0.65 : 0.3,
    has_password_breach:   signals.passwordExposed ? 1 : 0,
  };

  return {
    leakhunter:   lh,
    xposedornot:  xon,
    merged: {
      breachCount:   mergedBreachCount,
      exposures,
      signals,
      passwordLeaks,
    },
    features,
  };
}

module.exports = { gatherIntelligence };
