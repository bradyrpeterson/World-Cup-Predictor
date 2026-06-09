"""
Statalysts World Cup 2026 — round-by-round advancement model
=============================================================
Methodology (mirrors the Statalysts CFB/CBB stack):
  1. Power rating layer  -> margin-weighted Elo over 150 yrs of internationals
                            (the "Quality Score" analog)
  2. ML match layer      -> multinomial logistic regression (W/D/L) +
                            Poisson goal models, features = Elo diff, form,
                            home/host advantage
  3. Simulation layer    -> 20,000 Monte Carlo tournaments through the real
                            2026 bracket (12 groups, third-place allocation,
                            R32 -> Final) => P(reach round) for all 48 teams

Data: github.com/martj42/international_results (free, updated daily)
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor

RNG = np.random.default_rng(2026)
N_SIMS = 20000

# ----------------------------------------------------------------------------
# 1. ELO POWER RATINGS
# ----------------------------------------------------------------------------
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 50,
    "Copa América": 50, "UEFA Euro": 50, "African Cup of Nations": 50,
    "AFC Asian Cup": 50, "CONCACAF Championship": 50, "Gold Cup": 50,
    "UEFA Nations League": 40, "CONCACAF Nations League": 40,
    "Confederations Cup": 40,
}
DEFAULT_K = 30
FRIENDLY_K = 20
HOME_ELO = 80.0  # home advantage in Elo points


def margin_multiplier(gd: int) -> float:
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0  # 1.75 at 3 goals, +0.125 per extra goal


def run_elo(df: pd.DataFrame):
    elo, history = {}, []
    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh, ra = elo.get(h, 1500.0), elo.get(a, 1500.0)
        home_adv = 0.0 if row.neutral else HOME_ELO
        history.append((rh, ra, home_adv))
        diff = (rh + home_adv) - ra
        exp_h = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        if row.home_score > row.away_score:
            res = 1.0
        elif row.home_score == row.away_score:
            res = 0.5
        else:
            res = 0.0
        k = FRIENDLY_K if row.tournament == "Friendly" else \
            K_BY_TOURNAMENT.get(row.tournament, DEFAULT_K)
        delta = k * margin_multiplier(abs(int(row.home_score - row.away_score))) * (res - exp_h)
        elo[h], elo[a] = rh + delta, ra - delta
    pre = pd.DataFrame(history, columns=["elo_h", "elo_a", "home_adv"])
    return elo, pre


# ----------------------------------------------------------------------------
# 2. ML MATCH MODELS
# ----------------------------------------------------------------------------
def rolling_form(df: pd.DataFrame, window=10):
    """Mean goal differential over each team's previous `window` matches."""
    last = {}
    fh, fa = [], []
    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        gh = last.get(h, [])
        ga = last.get(a, [])
        fh.append(np.mean(gh[-window:]) if gh else 0.0)
        fa.append(np.mean(ga[-window:]) if ga else 0.0)
        gd = int(row.home_score - row.away_score)
        last.setdefault(h, []).append(gd)
        last.setdefault(a, []).append(-gd)
    return np.array(fh), np.array(fa), last


def build_models():
    df = pd.read_csv("results.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)

    elo_final, pre = run_elo(df)
    form_h, form_a, form_state = rolling_form(df)

    df["elo_diff"] = (pre["elo_h"] + pre["home_adv"]) - pre["elo_a"]
    df["form_diff"] = form_h - form_a

    train = df[df["date"] >= "2002-01-01"].copy()
    X = train[["elo_diff", "form_diff"]].values
    y = np.where(train.home_score > train.away_score, 2,
                 np.where(train.home_score == train.away_score, 1, 0))

    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)  # classes: 0=loss,1=draw,2=win

    # Scale features for the Poisson goal models (raw Elo diffs are O(100s)
    # and destabilize the solver). SCALE is baked into the saved models.
    SCALE = np.array([400.0, 2.0])
    Xs = X / SCALE
    pois_h = PoissonRegressor(alpha=1e-6, max_iter=3000).fit(Xs, train.home_score.values)
    pois_a = PoissonRegressor(alpha=1e-6, max_iter=3000).fit(-Xs, train.away_score.values)
    print("Poisson coefs:", pois_h.coef_, pois_a.coef_)

    acc = (clf.predict(X) == y).mean()
    print(f"Trained on {len(train):,} matches | in-sample W/D/L accuracy: {acc:.3f}")

    current_form = {t: np.mean(g[-10:]) if g else 0.0 for t, g in form_state.items()}
    return clf, pois_h, pois_a, elo_final, current_form


# ----------------------------------------------------------------------------
# 3. TOURNAMENT STRUCTURE (confirmed final field, Dec 2025 draw + Mar 2026 playoffs)
# ----------------------------------------------------------------------------
GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}
DISPLAY = {"Czech Republic": "Czechia", "Turkey": "Türkiye"}
HOSTS = {"United States", "Mexico", "Canada"}
HOST_ELO = 60.0  # partial home-crowd edge for the three co-hosts

# Round-of-32 bracket (FIFA match numbers). '1X'=group winner, '2X'=runner-up,
# '3'=third-place slot with its allowed source groups.
R32 = {
    73: ("2A", "2B"), 74: ("1E", ("3", "ABCDF")), 75: ("1F", "2C"),
    76: ("1C", "2F"), 77: ("1I", ("3", "CDFGH")), 78: ("2E", "2I"),
    79: ("1A", ("3", "CEFHI")), 80: ("1L", ("3", "EHIJK")),
    81: ("1D", ("3", "BEFIJ")), 82: ("1G", ("3", "AEHIJ")),
    83: ("2K", "2L"), 84: ("1H", "2J"), 85: ("1B", ("3", "EFGIJ")),
    86: ("1J", "2H"), 87: ("1K", ("3", "DEIJL")), 88: ("2D", "2G"),
}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
THIRD_SLOTS = [(m, set(spec[1][1])) for m, spec in R32.items()
               if isinstance(spec[1], tuple)]


# ----------------------------------------------------------------------------
# 4. SIMULATION
# ----------------------------------------------------------------------------
class Sim:
    def __init__(self, clf, pois_h, pois_a, elo, form):
        teams = [t for g in GROUPS.values() for t in g]
        self.elo = {t: elo[t] for t in teams}
        self.form = {t: form.get(t, 0.0) for t in teams}

        # Precompute every pairwise matchup once -> sim loop is pure lookups.
        eff = np.array([self.elo[t] + (HOST_ELO if t in HOSTS else 0.0)
                        for t in teams])
        frm = np.array([self.form[t] for t in teams])
        n = len(teams)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        X = np.column_stack([(eff[ii] - eff[jj]).ravel(),
                             (frm[ii] - frm[jj]).ravel()])
        Xs = X / np.array([400.0, 2.0])
        lam_a = np.clip(pois_h.predict(Xs), 0.15, 4.5).reshape(n, n)
        lam_b = np.clip(pois_a.predict(-Xs), 0.15, 4.5).reshape(n, n)
        proba = clf.predict_proba(X)               # cols: loss, draw, win
        p_win, p_draw, p_loss = proba[:, 2], proba[:, 1], proba[:, 0]
        ko = p_win + p_draw * (p_win / (p_win + p_loss))  # ET/pens by strength
        self.idx = {t: k for k, t in enumerate(teams)}
        self.lam_a, self.lam_b = lam_a, lam_b
        self.p_ko = ko.reshape(n, n)

    def sample_score(self, a, b):
        i, j = self.idx[a], self.idx[b]
        return RNG.poisson(self.lam_a[i, j]), RNG.poisson(self.lam_b[i, j])

    def knockout_winner(self, a, b):
        return a if RNG.random() < self.p_ko[self.idx[a], self.idx[b]] else b

    def play_group(self, teams):
        stats = {t: [0, 0, 0] for t in teams}     # pts, gd, gf
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = teams[i], teams[j]
                ga, gb = self.sample_score(a, b)
                stats[a][1] += ga - gb; stats[a][2] += ga
                stats[b][1] += gb - ga; stats[b][2] += gb
                if ga > gb:   stats[a][0] += 3
                elif gb > ga: stats[b][0] += 3
                else:         stats[a][0] += 1; stats[b][0] += 1
        order = sorted(teams, key=lambda t: (stats[t][0], stats[t][1],
                                             stats[t][2], RNG.random()),
                       reverse=True)
        return order, stats

    @staticmethod
    def assign_thirds(qualified_groups):
        """Backtracking match of 8 qualified third-place groups -> 8 slots."""
        slots = sorted(THIRD_SLOTS, key=lambda s: len(s[1] & qualified_groups))
        assign, used = {}, set()

        def bt(i):
            if i == len(slots):
                return True
            match, allowed = slots[i]
            for g in allowed & qualified_groups - used:
                used.add(g); assign[match] = g
                if bt(i + 1):
                    return True
                used.discard(g); assign.pop(match)
            return False

        bt(0)
        return assign

    def run_once(self, counts):
        winners, runners, third_rank = {}, {}, []
        for g, teams in GROUPS.items():
            order, stats = self.play_group(teams)
            winners[g], runners[g] = order[0], order[1]
            third_rank.append((g, order[2], stats[order[2]]))

        third_rank.sort(key=lambda x: (x[2][0], x[2][1], x[2][2], RNG.random()),
                        reverse=True)
        best8 = third_rank[:8]
        qual_groups = {g for g, _, _ in best8}
        third_team = {g: t for g, t, _ in best8}
        slot_map = self.assign_thirds(qual_groups)

        in_r32 = set(winners.values()) | set(runners.values()) | set(third_team.values())
        for t in in_r32:
            counts[t]["R32"] += 1

        def resolve(spec, match):
            if isinstance(spec, tuple):
                return third_team[slot_map[match]]
            kind, g = spec[0], spec[1]
            return winners[g] if kind == "1" else runners[g]

        alive = {}
        for m, (s1, s2) in R32.items():
            a, b = resolve(s1, m), resolve(s2, m)
            alive[m] = self.knockout_winner(a, b)
        for t in alive.values():
            counts[t]["R16"] += 1
        for rnd, stage in ((R16, "QF"), (QF, "SF"), (SF, "F")):
            nxt = {}
            for m, (m1, m2) in rnd.items():
                nxt[m] = self.knockout_winner(alive[m1], alive[m2])
            for t in nxt.values():
                counts[t][stage] += 1
            alive = nxt
        champ = self.knockout_winner(alive[101], alive[102])
        counts[champ]["W"] += 1


def main():
    clf, pois_h, pois_a, elo, form = build_models()
    sim = Sim(clf, pois_h, pois_a, elo, form)

    stages = ["R32", "R16", "QF", "SF", "F", "W"]
    counts = {t: dict.fromkeys(stages, 0) for g in GROUPS.values() for t in g}

    for i in range(N_SIMS):
        sim.run_once(counts)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,} sims done")

    out = []
    for g, teams in GROUPS.items():
        for t in teams:
            row = {"team": DISPLAY.get(t, t), "group": g,
                   "elo": round(sim.elo[t], 1)}
            row.update({s: round(100 * counts[t][s] / N_SIMS, 1) for s in stages})
            out.append(row)
    out.sort(key=lambda r: (-r["W"], -r["F"], -r["SF"], -r["elo"]))

    with open("wc2026_probs.json", "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n{'Team':<24}{'Elo':>7}{'R32':>7}{'R16':>7}{'QF':>7}{'SF':>7}{'Final':>7}{'Champ':>7}")
    for r in out[:20]:
        print(f"{r['team']:<24}{r['elo']:>7}{r['R32']:>7}{r['R16']:>7}"
              f"{r['QF']:>7}{r['SF']:>7}{r['F']:>7}{r['W']:>7}")


if __name__ == "__main__":
    main()