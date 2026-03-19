// ===== services/xonService.js =====
// XposedOrNot — free, no API key required
// GET https://api.xposedornot.com/v1/check-email/{email}
// 404 = no breaches (not an error)

const axios = require('axios');

async function checkXON(email) {
  try {
    const { data } = await axios.get(
      `https://api.xposedornot.com/v1/check-email/${encodeURIComponent(email)}`,
      {
        headers: { 'User-Agent': 'BreachIntel/3.0 (security-research)' },
        timeout: 10000,
      }
    );

    const breaches     = Array.isArray(data.breaches) ? data.breaches : [];
    const metrics      = data.BreachMetrics || {};

    // passwords_leaked can be a number OR an array — handle both
    let passwordLeakedCount = 0;
    if (Array.isArray(metrics.passwords_leaked)) {
      passwordLeakedCount = metrics.passwords_leaked.length;
    } else if (typeof metrics.passwords_leaked === 'number') {
      passwordLeakedCount = metrics.passwords_leaked;
    }

    return {
      breachCount:        breaches.length,
      breachNames:        breaches,
      passwordLeakCount:  passwordLeakedCount,
      risk:               metrics.risk || 'Unknown',
      raw:                data,
    };
  } catch (err) {
    if (err.response?.status === 404) {
      return { breachCount: 0, breachNames: [], passwordLeakCount: 0, risk: 'None', raw: null };
    }
    console.error('[XON] Error:', err.message);
    return { breachCount: 0, breachNames: [], passwordLeakCount: 0, risk: 'Error', raw: null };
  }
}

module.exports = { checkXON };
