#!/usr/bin/env python3
"""
data_pipeline.py — Breach Intel ML Data Pipeline
Reads raw feature dict from stdin (JSON), normalises via log1p, clamps floats.
Outputs normalised feature vector to stdout (JSON).
Node.js ↔ Python bridge: stdin → stdout, exit 0 = success, exit 1 = error.
"""

import sys
import json
import math


def log1p_norm(value, cap):
    """Log1p normalisation with cap: log(1+v) / log(1+cap)"""
    try:
        v = float(value)
        return math.log1p(max(v, 0)) / math.log1p(cap)
    except Exception:
        return 0.0


def clamp(value, lo=0.0, hi=1.0):
    """Clamp value to [lo, hi], NaN → lo"""
    try:
        v = float(value)
        if math.isnan(v):
            return lo
        return max(lo, min(hi, v))
    except Exception:
        return lo


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "Empty input"}))
        sys.exit(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON parse error: {e}"}))
        sys.exit(1)

    # Support both { features: {...} } and flat dict
    if "features" in payload:
        f = payload["features"]
    else:
        f = payload

    # ── Normalise count features via log1p ───────────────────────────────────
    breach_norm      = log1p_norm(f.get("breach_count", 0),       200)   # cap 200 breaches
    password_norm    = log1p_norm(f.get("password_leaks", 0),     100)   # cap 100 password leaks
    severity_norm    = clamp(float(f.get("avg_severity", 0)) / 10.0)     # 0–10 → 0–1
    critical_norm    = log1p_norm(f.get("critical_count", 0),     50)    # cap 50 critical
    recent_norm      = log1p_norm(f.get("recent_breaches", 0),    30)    # cap 30 recent

    # ── Clamp float signals ──────────────────────────────────────────────────
    login_anomaly    = clamp(f.get("login_anomaly_score",  0.0))
    public_exposure  = clamp(f.get("public_exposure",      0.0))
    social_risk      = clamp(f.get("social_risk_score",    0.0))
    has_password     = clamp(f.get("has_password_breach",  0))

    # ── Danger composite ─────────────────────────────────────────────────────
    danger_composite = clamp(
        breach_norm * 0.30 +
        password_norm * 0.25 +
        critical_norm * 0.25 +
        severity_norm * 0.20
    )

    output = {
        # Normalised features (passed to model.py)
        "breach_norm":         round(breach_norm,     4),
        "password_norm":       round(password_norm,   4),
        "severity_norm":       round(severity_norm,   4),
        "critical_norm":       round(critical_norm,   4),
        "recent_norm":         round(recent_norm,     4),
        "login_anomaly_score": round(login_anomaly,   4),
        "public_exposure":     round(public_exposure, 4),
        "social_risk_score":   round(social_risk,     4),
        "has_password_breach": round(has_password,    4),
        "danger_composite":    round(danger_composite,4),
        # Raw counts (for display in UI)
        "_raw_breach_count":   f.get("breach_count", 0),
        "_raw_password_leaks": f.get("password_leaks", 0),
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
