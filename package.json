// ===== config.js =====
// Uses JS getters so env vars are read LIVE on every access
// (Fixes the "frozen at require() time" bug)

const config = {
  get PORT()                { return process.env.PORT || 3000; },
  get LEAKHUNTER_API_KEY()  { return process.env.LEAKHUNTER_API_KEY || ''; },
  get LEAKHUNTER_API_HOST() { return process.env.LEAKHUNTER_API_HOST || 'leakhunter-ai1.p.rapidapi.com'; },
  get PYTHON_CMD()          { return process.env.PYTHON_CMD || 'python3'; },
  get ALLOWED_ORIGIN()      { return process.env.ALLOWED_ORIGIN || '*'; },
  get DATA_MODE()           { return process.env.DATA_MODE || 'live'; },
};

module.exports = config;
