// ===== server.js =====
// Load .env from project ROOT (one level up from backend/)
// This MUST be first — before any other require
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

// Debug: confirm key loaded (remove after testing)
console.log('[ENV] LEAKHUNTER_API_KEY loaded:', process.env.LEAKHUNTER_API_KEY ? '✓ YES' : '✗ MISSING');

const express   = require('express');
const cors      = require('cors');
const rateLimit = require('express-rate-limit');
const intelRoutes = require('./routes/intelRoutes');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({ origin: process.env.ALLOWED_ORIGIN || '*' }));
app.use(express.json());

// Security headers
app.use((req, res, next) => {
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Permissions-Policy', 'geolocation=()');
  next();
});

// Rate limiting: 40 req/min per IP
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 40,
  message: { error: 'Too many requests. Please wait 1 minute.' }
});
app.use('/api/', limiter);

// ── Static frontend ───────────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// ── API Routes ────────────────────────────────────────────────────────────────
app.use('/api', intelRoutes);

// ── Catch-all: serve index.html ───────────────────────────────────────────────
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'frontend', 'index.html'));
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n╔══════════════════════════════════════╗`);
  console.log(`║   BREACH INTEL · v3.0  ONLINE        ║`);
  console.log(`║   http://localhost:${PORT}              ║`);
  console.log(`╚══════════════════════════════════════╝\n`);
});
