// ===== routes/intelRoutes.js =====
const express        = require('express');
const router         = express.Router();
const intelController = require('../controllers/intelController');

// POST /api/analyze  — main scan endpoint
router.post('/analyze', (req, res) => {
  const { identifier } = req.body;

  // Input validation
  if (!identifier || typeof identifier !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid identifier.' });
  }
  const clean = identifier.trim().replace(/[\x00-\x1f]/g, '');
  if (clean.length === 0 || clean.length > 254) {
    return res.status(400).json({ error: 'Identifier must be 1–254 characters.' });
  }
  const emailRx = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRx.test(clean)) {
    return res.status(400).json({ error: 'Invalid email format.' });
  }

  req.body.identifier = clean;
  intelController.getIntel(req, res);
});

// GET /api/health  — system status check
router.get('/health', intelController.health);

module.exports = router;
