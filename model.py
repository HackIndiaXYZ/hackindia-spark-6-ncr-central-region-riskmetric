// ===== services/leakhunterService.js =====
// LeakHunter AI via RapidAPI
//
// Request (from request.txt):
//   POST /leakhunt HTTP/1.1
//   X-Rapidapi-Key: <key>
//   X-Rapidapi-Host: leakhunter-ai1.p.rapidapi.com
//   Content-Type: application/json
//   Host: leakhunter-ai1.p.rapidapi.com
//
//   {"email":"user@example.com"}

const axios = require('axios');
const path  = require('path');

// Always load .env from project root regardless of cwd
require('dotenv').config({ path: path.join(__dirname, '..', '..', '.env') });

const LEAKHUNTER_HOST = 'leakhunter-ai1.p.rapidapi.com';
const LEAKHUNTER_URL  = `https://${LEAKHUNTER_HOST}/leakhunt`;

async function checkLeakHunter(email) {
  // Read key live every call (getter pattern — never frozen)
  const apiKey = process.env.LEAKHUNTER_API_KEY || '';

  console.log(`[LeakHunter] API key present: ${apiKey ? 'YES (' + apiKey.substring(0,8) + '...)' : 'NO — check .env'}`);

  if (!apiKey) {
    console.warn('[LeakHunter] No API key — set LEAKHUNTER_API_KEY in your .env file');
    return { isExposed: false, exposureCount: 0, riskScore: 0, riskLevel: 'unknown', exposures: [], signals: {}, _skipped: true };
  }

  console.log(`[LeakHunter] Querying for: ${email}`);

  try {
    const response = await axios.post(
      LEAKHUNTER_URL,
      { email },
      {
        headers: {
          'Content-Type':    'application/json',
          'X-Rapidapi-Key':  apiKey,
          'X-Rapidapi-Host': LEAKHUNTER_HOST,
        },
        timeout: 15000,
      }
    );

    console.log(`[LeakHunter] Response status: ${response.status}`);
    return normaliseLeakHunterResponse(response.data);

  } catch (err) {
    if (err.response?.status === 404) {
      console.log('[LeakHunter] 404 — no breaches found (clean email)');
      return { isExposed: false, exposureCount: 0, riskScore: 0, riskLevel: 'clean', exposures: [], signals: {} };
    }
    console.error('[LeakHunter] Error:', err.response?.status, err.message);
    throw new Error(`LeakHunter API error ${err.response?.status || ''}: ${err.message}`);
  }
}

function normaliseLeakHunterResponse(raw) {
  const d = typeof raw === 'string' ? JSON.parse(raw) : raw;

  const exposures = Array.isArray(d.exposures) ? d.exposures.map(e => ({
    name:             e.name             || 'Unknown',
    type:             e.type             || 'credential',
    date:             e.date             || null,
    passwordIncluded: Boolean(e.passwordIncluded),
    severity:         typeof e.severity === 'number' ? e.severity : 5,
  })) : [];

  const signals = d.signals || {
    passwordExposed:    exposures.some(e => e.passwordIncluded),
    recentBreach:       exposures.some(e => e.date && parseInt(e.date.substring(0, 4)) >= 2020),
    multipleExposures:  exposures.length > 1,
    highSeverityBreach: exposures.some(e => e.severity >= 9),
  };

  return {
    isExposed:     Boolean(d.isExposed),
    exposureCount: typeof d.exposureCount === 'number' ? d.exposureCount : exposures.length,
    riskScore:     typeof d.riskScore     === 'number' ? d.riskScore : 0,
    riskLevel:     d.riskLevel || 'unknown',
    exposures,
    signals,
  };
}

module.exports = { checkLeakHunter };
