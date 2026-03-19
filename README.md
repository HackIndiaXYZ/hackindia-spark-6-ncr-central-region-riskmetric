# BREACH INTEL — Ultimate Edition v3.0
### AI-Powered Cyber Exposure Intelligence Platform

```
Browser → Node.js → Python ML → RandomForest + SHAP → Response
```

---

## 🔌 APIs Used

| API | Auth | Endpoint |
|-----|------|----------|
| **LeakHunter AI** | RapidAPI Key | `POST https://leakhunter-ai1.p.rapidapi.com/leakhunt` |
| **XposedOrNot** | None (free) | `GET https://api.xposedornot.com/v1/check-email/{email}` |

### LeakHunter AI Request
```bash
curl --request POST \
  --url https://leakhunter-ai1.p.rapidapi.com/leakhunt \
  --header 'Content-Type: application/json' \
  --header 'x-rapidapi-host: leakhunter-ai1.p.rapidapi.com' \
  --header 'x-rapidapi-key: YOUR_KEY' \
  --data '{"email":"user@example.com"}'
```

### LeakHunter AI Response Shape
```json
{
  "isExposed": true,
  "exposureCount": 170,
  "riskScore": 0.98,
  "riskLevel": "critical",
  "exposures": [
    { "name": "Adobe", "type": "credential", "date": "2013-10-04", "passwordIncluded": true, "severity": 9 }
  ],
  "signals": {
    "passwordExposed": true,
    "recentBreach": true,
    "multipleExposures": true,
    "highSeverityBreach": true
  }
}
```

---

## 🗂️ Project Structure

```
breachintel/
├── frontend/
│   └── index.html          ← All UI (HTML5 + CSS + Vanilla JS)
├── backend/
│   ├── server.js           ← Express entry point
│   ├── package.json
│   ├── config/config.js    ← JS getter pattern (live env vars)
│   ├── routes/intelRoutes.js
│   ├── controllers/intelController.js
│   └── services/
│       ├── leakhunterService.js   ← LeakHunter AI (RapidAPI)
│       ├── xonService.js          ← XposedOrNot (free)
│       └── intelligenceService.js ← Merges both, derives features
├── ml/
│   ├── data_pipeline.py    ← log1p normalisation + clamping
│   └── model.py            ← RandomForest + IsolationForest + SHAP
├── .env.example
└── README.md
```

---

## 🚀 Setup

### Prerequisites
- Node.js v18+
- Python 3.8+

### 1. Install Python dependencies
```bash
pip install scikit-learn numpy joblib shap
```

### 2. Install Node.js dependencies
```bash
cd backend
npm install
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and set LEAKHUNTER_API_KEY
```

### 4. Start the server
```bash
cd backend
npm start
```

Open: http://localhost:3000

---

## 🧠 ML Pipeline

```
LeakHunter AI Response
        ↓
  Extract features:
  breach_count, password_leaks, avg_severity,
  critical_count, recent_breaches, signals...
        ↓
  data_pipeline.py
  log1p normalisation → 9 normalised features
        ↓
  model.py
  RandomForest (200 trees) + IsolationForest
  → class probabilities [p_low, p_medium, p_high]
  score = p_low×18 + p_medium×52 + p_high×86
  + anomaly boost if IsoForest flags it
        ↓
  SHAP TreeExplainer
  → per-feature contribution scores
        ↓
  Response: { score, risk_level, factors, shap_factors }
```

### 9 Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `breach_norm` | 0–1 | log1p normalised breach count (cap=200) |
| `password_norm` | 0–1 | log1p normalised password leak count (cap=100) |
| `severity_norm` | 0–1 | Average severity / 10 |
| `critical_norm` | 0–1 | log1p normalised critical breach count |
| `recent_norm` | 0–1 | log1p normalised recent breach count (≥2020) |
| `login_anomaly_score` | 0–1 | Derived from signals.recentBreach |
| `public_exposure` | 0–1 | breach_count / 200 (clamped) |
| `social_risk_score` | 0–1 | Derived from signals.multipleExposures |
| `has_password_breach` | 0/1 | Binary: signals.passwordExposed |

---

## 🔐 Security Features

- Rate limiting: 40 req/min per IP
- Security headers (X-Frame-Options, CSP, etc.)
- Email input validation + sanitisation
- No shell injection risk (JSON stdin/stdout bridge)
- API key only in .env (never sent to frontend)
- CORS restricted to ALLOWED_ORIGIN

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze` | Main scan: `{ identifier: "email@example.com" }` |
| `GET` | `/api/health` | System status + config check |

---

## 🔮 Future Roadmap

- Shodan API integration (open ports / exposed servers)
- Redis caching (24h per email)
- PostgreSQL breach history storage
- JWT auth + scan history
- Docker deployment
- PyTorch neural network on real data
- Target: >90% accuracy on real-world breach classification

---

Built for learning · security awareness · open-source development
March 2026 · v3.0
