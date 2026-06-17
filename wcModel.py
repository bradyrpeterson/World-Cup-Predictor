"""
Statalysts → Jump Trading Probability Cup Bot
==============================================
Fully automated. Run once, predicts everything.

Usage:
    python bot.py                   # run full prediction cycle
    python bot.py --dry-run         # show predictions without submitting
    python bot.py --results         # show your settled scores + Brier breakdown
    python bot.py --status          # show open predictions still to settle

Setup (one time only):
    pip install requests anthropic numpy pandas scikit-learn
    export SPORTSPREDICT_KEY=sp_live_...
    export ANTHROPIC_API_KEY=sk-ant-...   # optional — used for edge markets
"""

import os
import sys
import math
import time
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import requests
import re
from datetime import datetime, timezone
from io import StringIO

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

API        = "https://api.sportspredict.com/api/v1"
SP_KEY     = os.environ.get("SPORTSPREDICT_KEY", "")
ANT_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
DRY_RUN    = False   # overridden by --dry-run flag
BATCH_SIZE = 50
RATE_SLEEP = 1.1     # seconds between batched requests (stay under 60/min)

HEADERS = {
    "Authorization": f"Bearer {SP_KEY}",
    "Content-Type":  "application/json",
}

# ─── Elo / ML model constants (mirrors wc_model.py) ──────────────────────────

K          = 35        # Elo K-factor
HOME_ELO   = 100       # host advantage in Elo points
ELO_START  = 1500
FORM_W     = 0.15      # exponential smoothing weight for recent-form feature
SCALE_ELO  = 400       # Elo diff scaling for Poisson features
MIN_LAM    = 0.20
MAX_LAM    = 4.5

# 2026 World Cup host nations
HOSTS = {"USA", "Mexico", "Canada"}

# Official 2026 World Cup groups
GROUPS = {
    "A": ["Mexico","Bolivia","Ecuador","Uruguay"],
    "B": ["Germany","Japan","Australia","Chile"],
    "C": ["Argentina","South Africa","Morocco","Iraq"],
    "D": ["Spain","Brazil","Japan","Egypt"],   # placeholder if needed
    "E": ["France","USA","Panama","Algeria"],
    "F": ["England","Senegal","Cameroon","Serbia"],
    "G": ["Portugal","Croatia","Colombia","New Zealand"],
    "H": ["Netherlands","South Korea","Poland","Saudi Arabia"],
    "I": ["Belgium","Mexico","Venezuela","Cuba"],
    "J": ["Brazil","Switzerland","Ukraine","Peru"],
    "K": ["Argentina","USA","Canada","Slovenia"],
    "L": ["Spain","England","Morocco","Australia"],
}

# ─── 1. DATA: pull martj42 international results ─────────────────────────────

def fetch_match_data() -> pd.DataFrame:
    print("📥  Fetching international match history …")
    url = (
        "https://raw.githubusercontent.com/martj42/international_results"
        "/master/results.csv"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), parse_dates=["date"])
        # keep meaningful competitive & friendly matches
        df = df[df["date"] >= "2002-01-01"].copy()
        df = df.dropna(subset=["home_score", "away_score"])
        print(f"   ✓ {len(df):,} matches loaded (2002–present)")
        return df
    except Exception as e:
        print(f"   ✗ Could not fetch data: {e}")
        sys.exit(1)

# ─── 2. ELO ratings ──────────────────────────────────────────────────────────

def build_elo_ratings(df: pd.DataFrame) -> tuple[dict, dict]:
    """Margin-weighted Elo + exponential recent-form tracker."""
    elo  = {}
    form = {}

    def get(d: dict, team: str, default: float) -> float:
        return d.get(team, default)

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        neutral = bool(row.get("neutral", False))
        host_bonus = 0 if neutral else HOME_ELO

        eh = get(elo, h, ELO_START) + host_bonus
        ea = get(elo, a, ELO_START)

        expected_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        actual_h   = 0.5 if hs == as_ else (1.0 if hs > as_ else 0.0)

        # margin multiplier: cap at 3 goals
        margin = min(abs(hs - as_), 3)
        k_adj  = K * (1 + margin * 0.25)

        delta = k_adj * (actual_h - expected_h)

        elo[h]  = get(elo, h,  ELO_START) + delta
        elo[a]  = get(elo, a,  ELO_START) - delta

        form[h] = get(form, h, 0.0) * (1 - FORM_W) + actual_h   * FORM_W
        form[a] = get(form, a, 0.0) * (1 - FORM_W) + (1 - actual_h) * FORM_W

    return elo, form

# ─── 3. TRAIN ML models ──────────────────────────────────────────────────────

def build_models(df: pd.DataFrame, elo: dict, form: dict):
    """
    Train:
      clf    — logistic regression W/D/L classifier
      pois_h — Poisson home goals
      pois_a — Poisson away goals
    Features: [elo_diff/SCALE, form_diff]
    """
    from sklearn.linear_model import LogisticRegression, PoissonRegressor

    rows = []
    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]
        if h not in elo or a not in elo:
            continue
        neutral = bool(r.get("neutral", False))
        hb      = 0 if neutral else HOME_ELO
        ed      = (elo[h] + hb - elo[a]) / SCALE_ELO
        fd      = form.get(h, 0.5) - form.get(a, 0.5)
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        outcome = 2 if hs > as_ else (1 if hs == as_ else 0)  # 2=win,1=draw,0=loss
        rows.append([ed, fd, hs, as_, outcome])

    data   = pd.DataFrame(rows, columns=["elo_diff","form_diff","hg","ag","outcome"])
    X      = data[["elo_diff","form_diff"]].values
    y      = data["outcome"].values
    hg     = data["hg"].values
    ag     = data["ag"].values

    clf    = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    pois_h = PoissonRegressor(max_iter=500).fit(X, hg)
    pois_a = PoissonRegressor(max_iter=500).fit(-X, ag)

    return clf, pois_h, pois_a

# ─── 4. MATCH PROBABILITY ENGINE ─────────────────────────────────────────────

class MatchPredictor:
    """Wraps trained models to answer any binary question about a match."""

    def __init__(self, clf, pois_h, pois_a, elo: dict, form: dict):
        self.clf    = clf
        self.pois_h = pois_h
        self.pois_a = pois_a
        self.elo    = elo
        self.form   = form

    def _features(self, home: str, away: str) -> np.ndarray:
        eh  = self.elo.get(home, ELO_START)
        ea  = self.elo.get(away, ELO_START)
        # add host boost if applicable
        if home in HOSTS:
            eh += HOME_ELO
        ed  = (eh - ea) / SCALE_ELO
        fd  = self.form.get(home, 0.5) - self.form.get(away, 0.5)
        return np.array([[ed, fd]])

    def win_draw_loss(self, home: str, away: str) -> tuple[float, float, float]:
        """Returns (p_home_win, p_draw, p_away_win)."""
        X     = self._features(home, away)
        proba = self.clf.predict_proba(X)[0]   # [loss, draw, win]
        return proba[2], proba[1], proba[0]

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        X  = self._features(home, away)
        gh = float(np.clip(self.pois_h.predict(X)[0], MIN_LAM, MAX_LAM))
        ga = float(np.clip(self.pois_a.predict(-X)[0], MIN_LAM, MAX_LAM))
        return gh, ga

    def btts_prob(self, home: str, away: str) -> float:
        """Both teams to score: P(hg≥1) × P(ag≥1) via Poisson."""
        gh, ga = self.expected_goals(home, away)
        p_h_scores = 1 - math.exp(-gh)
        p_a_scores = 1 - math.exp(-ga)
        return p_h_scores * p_a_scores

    def over_goals_prob(self, home: str, away: str, line: float = 2.5) -> float:
        """P(total goals > line) by convolving two Poisson distributions."""
        gh, ga = self.expected_goals(home, away)
        threshold = int(math.floor(line))
        # P(total ≤ threshold) = sum over all combos
        p_under = 0.0
        for hg in range(threshold + 1):
            for ag in range(threshold + 1 - hg):
                ph = math.exp(-gh) * (gh ** hg) / math.factorial(hg)
                pa = math.exp(-ga) * (ga ** ag) / math.factorial(ag)
                p_under += ph * pa
        return 1 - p_under

    def clean_sheet_prob(self, home: str, away: str) -> float:
        """P(away scores 0) — home team clean sheet."""
        _, ga = self.expected_goals(home, away)
        return math.exp(-ga)

# ─── 5. QUESTION PARSER: map market question → probability ───────────────────

def parse_teams_from_match(match_name: str) -> tuple[str, str]:
    """'Mexico vs South Africa' → ('Mexico', 'South Africa')"""
    parts = re.split(r"\s+vs\.?\s+", match_name, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", ""

def predict_market(
    question: str,
    match_name: str,
    predictor: MatchPredictor,
) -> int | None:
    """
    Parse the binary question and return a 1-99 integer probability.
    Returns None if the question can't be answered by the model alone
    (those will be sent to Claude).
    """
    q    = question.lower().strip()
    home, away = parse_teams_from_match(match_name)
    if not home:
        return None

    # Normalize team name aliases (common FIFA vs casual names)
    ALIASES = {
        "united states": "USA",
        "u.s.": "USA",
        "south korea": "South Korea",
        "republic of ireland": "Republic of Ireland",
        "ivory coast": "Ivory Coast",
        "côte d'ivoire": "Ivory Coast",
        "democratic republic of congo": "DR Congo",
        "cape verde islands": "Cape Verde",
    }
    def norm(t):
        return ALIASES.get(t.lower(), t)

    home = norm(home)
    away = norm(away)

    pw, pd_, pl = predictor.win_draw_loss(home, away)

    # ── Win / loss patterns ──
    if re.search(rf"will {re.escape(home.lower())} win", q):
        return clamp(pw)
    if re.search(rf"will {re.escape(away.lower())} win", q):
        return clamp(pl)
    if re.search(r"will (the match|the game) end in a draw", q):
        return clamp(pd_)
    if re.search(r"draw|tie", q) and "regulation" not in q:
        return clamp(pd_)

    # Win in regulation specifically
    if re.search(r"win.*regulation", q):
        if re.search(rf"{re.escape(home.lower())}", q):
            return clamp(pw)
        if re.search(rf"{re.escape(away.lower())}", q):
            return clamp(pl)

    # ── Goals patterns ──
    over_m = re.search(r"over (\d+\.?\d*) goals?", q)
    if over_m:
        line = float(over_m.group(1))
        return clamp(predictor.over_goals_prob(home, away, line))

    under_m = re.search(r"under (\d+\.?\d*) goals?", q)
    if under_m:
        line = float(under_m.group(1))
        return clamp(1 - predictor.over_goals_prob(home, away, line))

    # Total goals shorthand
    if re.search(r"more than 2[. ]?5 goals", q) or re.search(r"2[. ]?5\+ goals", q):
        return clamp(predictor.over_goals_prob(home, away, 2.5))
    if re.search(r"at least 3 goals", q):
        return clamp(predictor.over_goals_prob(home, away, 2.5))

    # Both teams to score
    if re.search(r"both teams? (to )?score", q) or re.search(r"\bbtts\b", q):
        return clamp(predictor.btts_prob(home, away))

    # Clean sheet
    if re.search(r"clean sheet", q):
        if re.search(rf"{re.escape(home.lower())}", q):
            return clamp(predictor.clean_sheet_prob(home, away))
        if re.search(rf"{re.escape(away.lower())}", q):
            return clamp(predictor.clean_sheet_prob(away, home))

    # First half / half-time win (rough: home win prob × 0.7 since leads don't = HT wins)
    if re.search(r"(half.time|half time|first half).*win", q):
        if re.search(rf"{re.escape(home.lower())}", q):
            return clamp(pw * 0.70)
        if re.search(rf"{re.escape(away.lower())}", q):
            return clamp(pl * 0.70)

    # Anytime draw probability
    if "draw" in q:
        return clamp(pd_)

    return None   # hand off to Claude


def clamp(p: float) -> int:
    return max(1, min(99, round(p * 100)))

# ─── 6. CLAUDE fallback for unrecognised market questions ────────────────────

def claude_predict(question: str, match_name: str, home_elo: float, away_elo: float) -> int:
    """Ask Claude to estimate a probability for questions outside our model."""
    if not ANT_KEY:
        return 50   # neutral fallback if no key

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANT_KEY)
        prompt = f"""You are a precise sports probability estimator for the 2026 FIFA World Cup.

Match: {match_name}
Home team Elo rating: {home_elo:.0f}
Away team Elo rating: {away_elo:.0f}
(Average Elo ≈ 1500. Higher = stronger.)

Question: "{question}"

This is a binary (yes/no) question. Estimate the probability it resolves YES.
Consider team strength (reflected in Elo), typical World Cup match patterns,
and any domain knowledge about this specific question type.

Reply with ONLY a single integer between 1 and 99 (inclusive). Nothing else."""

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        val = int(re.sub(r"\D", "", raw))
        return max(1, min(99, val))

    except Exception as e:
        print(f"      Claude error ({e}) — using 50")
        return 50

# ─── 7. SPORTSPREDICT API helpers ────────────────────────────────────────────

def sp_get(path: str, params: dict = None) -> list | dict:
    resp = requests.get(f"{API}/{path}", headers=HEADERS, params=params or {})
    if resp.status_code == 401:
        print("✗ Invalid API key. Set SPORTSPREDICT_KEY and retry.")
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()

def sp_post(path: str, body: dict) -> dict:
    resp = requests.post(f"{API}/{path}", headers=HEADERS, json=body)
    resp.raise_for_status()
    return resp.json()

def sp_patch(path: str, body: dict) -> dict:
    resp = requests.patch(f"{API}/{path}", headers=HEADERS, json=body)
    resp.raise_for_status()
    return resp.json()

def bootstrap() -> tuple[str, str]:
    """Find the Probability Cup event and lobby, auto-join if needed."""
    print("🔍  Finding Probability Cup …")
    events = sp_get("events")
    event  = next((e for e in events if e.get("type") == "probability"), None)
    if not event:
        print("✗ No active Probability Cup event found.")
        sys.exit(1)
    print(f"   ✓ Event: {event['title']} (id: {event['id'][:8]}…)")

    lobbies = sp_get("lobbies", {"event_id": event["id"]})
    if not lobbies:
        print("✗ No lobby found for this event.")
        sys.exit(1)
    lobby = lobbies[0]

    if not lobby.get("joined"):
        print("   Joining lobby …")
        sp_post(f"lobbies/{lobby['id']}/join", {})
        print("   ✓ Joined")
    else:
        print(f"   ✓ Lobby: {lobby['name']} (already joined)")

    return event["id"], lobby["id"]

def fetch_all_markets(event_id: str, lobby_id: str) -> list[dict]:
    """Fetch all open markets, one match at a time (avoids >token limits)."""
    print("📋  Fetching open markets …")
    matches  = sp_get("matches", {"event_id": event_id, "lobby_id": lobby_id})
    print(f"   ✓ {len(matches)} matches with open markets")

    all_markets = []
    for match in matches:
        mkts = sp_get("markets", {"lobby_id": lobby_id, "match_id": match["id"]})
        for m in mkts:
            m["_match_name"] = match["name"]
            m["_match_id"]   = match["id"]
        all_markets.extend(mkts)
        time.sleep(0.05)   # gentle rate limiting

    print(f"   ✓ {len(all_markets)} open markets total")
    return all_markets

def already_predicted(lobby_id: str) -> set[str]:
    """Return set of market_ids already predicted (to skip dupes)."""
    preds = sp_get("predictions", {"lobby_id": lobby_id})
    return {p["market_id"] for p in preds}

def submit_batch(predictions: list[dict]) -> tuple[int, int]:
    """Submit up to 50 predictions, return (succeeded, failed)."""
    if not predictions:
        return 0, 0
    resp = sp_post("predictions/batch", {"predictions": predictions})
    for r in resp.get("results", []):
        if not r.get("success"):
            print(f"      ✗ market {r['market_id'][:8]}…: {r.get('error')}")
    return resp.get("succeeded", 0), resp.get("failed", 0)

# ─── 8. RESULTS / STATUS commands ────────────────────────────────────────────

def show_results(lobby_id: str):
    results = sp_get("results", {"lobby_id": lobby_id})
    if not results:
        print("No settled predictions yet.")
        return

    total_brier = sum(r["brier_score"] for r in results if r["brier_score"] is not None)
    avg_brier   = total_brier / len(results)

    print(f"\n{'─'*72}")
    print(f"  SETTLED RESULTS  ({len(results)} predictions)")
    print(f"{'─'*72}")
    for r in sorted(results, key=lambda x: x.get("brier_score") or 0):
        bs = r.get("brier_score")
        bs_str = f"{bs:.4f}" if bs is not None else "pending"
        print(f"  {r['question'][:55]:<55}  p={r['probability_submitted']:>2}  Brier={bs_str}")
    print(f"{'─'*72}")
    print(f"  Avg Brier: {avg_brier:.4f}  (crowd avg ~0.25 — lower is better)")
    print(f"{'─'*72}\n")

def show_status(lobby_id: str):
    preds = sp_get("predictions", {"lobby_id": lobby_id})
    open_ = [p for p in preds if p.get("market_status") == "open"]
    print(f"\n{len(open_)} open predictions still to settle:\n")
    for p in open_:
        print(f"  {p['question'][:60]:<60}  p={p['probability']:>2}")
    print()

# ─── 9. MAIN LOOP ─────────────────────────────────────────────────────────────

def main():
    global DRY_RUN

    parser = argparse.ArgumentParser(description="Statalysts Probability Cup Bot")
    parser.add_argument("--dry-run",  action="store_true", help="Print predictions without submitting")
    parser.add_argument("--results",  action="store_true", help="Show settled results & Brier scores")
    parser.add_argument("--status",   action="store_true", help="Show open predictions")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    if not SP_KEY:
        print("✗  SPORTSPREDICT_KEY not set. Run: export SPORTSPREDICT_KEY=sp_live_...")
        sys.exit(1)

    # ── Bootstrap ──
    event_id, lobby_id = bootstrap()

    if args.results:
        show_results(lobby_id)
        return
    if args.status:
        show_status(lobby_id)
        return

    # ── Train models ──
    print("\n🧠  Training Elo + ML models …")
    df            = fetch_match_data()
    elo, form     = build_elo_ratings(df)
    clf, ph, pa   = build_models(df, elo, form)
    predictor     = MatchPredictor(clf, ph, pa, elo, form)
    print(f"   ✓ Models trained on {len(df):,} matches")

    # ── Fetch markets ──
    markets      = fetch_all_markets(event_id, lobby_id)
    skip_ids     = already_predicted(lobby_id)
    new_markets  = [m for m in markets if m["id"] not in skip_ids]
    print(f"   {len(skip_ids)} already predicted — {len(new_markets)} new markets to predict\n")

    # ── Score each market ──
    print("🎯  Generating predictions …\n")
    predictions  = []
    model_count  = 0
    claude_count = 0

    for m in new_markets:
        match_name = m["_match_name"]
        question   = m["question"]

        prob = predict_market(question, match_name, predictor)
        source = "MODEL"

        if prob is None:
            home, away = parse_teams_from_match(match_name)
            h_elo = elo.get(home, ELO_START)
            a_elo = elo.get(away, ELO_START)
            prob  = claude_predict(question, match_name, h_elo, a_elo)
            source = "CLAUDE"
            claude_count += 1
        else:
            model_count += 1

        tag = "🤖" if source == "CLAUDE" else "📊"
        print(f"  {tag} [{source:6}] {match_name:<30}  p={prob:>2}  \"{question[:50]}\"")

        predictions.append({
            "market_id":  m["id"],
            "lobby_id":   lobby_id,
            "probability": prob,
        })

    print(f"\n  Model: {model_count} | Claude: {claude_count} | Total: {len(predictions)}")

    # ── Submit ──
    if DRY_RUN:
        print("\n⚠️   DRY RUN — nothing submitted.")
        return

    if not predictions:
        print("\n✅  Nothing new to submit — all markets already predicted.")
        return

    print(f"\n📤  Submitting {len(predictions)} predictions in batches of {BATCH_SIZE} …")
    total_ok = 0
    total_fail = 0

    for i in range(0, len(predictions), BATCH_SIZE):
        chunk = predictions[i:i+BATCH_SIZE]
        ok, fail = submit_batch(chunk)
        total_ok   += ok
        total_fail += fail
        batch_num   = i // BATCH_SIZE + 1
        print(f"   Batch {batch_num}: {ok}/{len(chunk)} succeeded")
        if i + BATCH_SIZE < len(predictions):
            time.sleep(RATE_SLEEP)

    print(f"\n✅  Done.  {total_ok} submitted, {total_fail} failed.")
    print(f"   Check your leaderboard at: https://sportspredict.com/probabilitycup\n")


if __name__ == "__main__":
    main()