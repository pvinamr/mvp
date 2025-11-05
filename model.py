import pandas as pd
import numpy as np
from nfl_data_py import import_pbp_data, import_schedules
import math

# ================== CONFIG ==================
SEASON = 2025
WEEK   = 8
WP_MIN, WP_MAX = 0.07, 0.93
N0 = 175                 # shrinkage prior (in plays)
TRAIN_YEARS_BACK = 3     # use last 3 seasons for learning
# ============================================

# ---------- helpers ----------
def bool_series(df, col):
    return df.get(col, pd.Series(False, index=df.index)).fillna(False).astype(bool)

def soft_clip(series: pd.Series, c: float) -> pd.Series:
    return c * np.tanh(series / c)

def build_team_ratings(pbp_subset: pd.DataFrame):
    """Build shrunken EPA (off/def), pace (off/def per-game), and Eckel (off/def) from a subset (no leakage)."""
    if pbp_subset.empty:
        return None

    kneel_s = bool_series(pbp_subset, "qb_kneel")
    spike_s = bool_series(pbp_subset, "qb_spike")

    # ---- filtered snaps for EPA (same rules you use)
    df = pbp_subset.loc[
        (pbp_subset["play_type"].isin(["pass","run"]))
        & (pbp_subset["epa"].notna())
        & (~kneel_s) & (~spike_s)
        & (pbp_subset["wp"].between(WP_MIN, WP_MAX, inclusive="both"))
    ].copy()

    if df.empty:
        return None

    # ---- Off/Def EPA
    off_rank = (df.groupby("posteam", observed=True)
                  .agg(plays=("epa","size"), epa_per_play=("epa","mean"))
                  .reset_index().rename(columns={"posteam":"team"}))
    def_rank = (df.groupby("defteam", observed=True)
                  .agg(plays_against=("epa","size"), epa_allowed=("epa","mean"))
                  .reset_index().rename(columns={"defteam":"team"}))

    # league means for shrink
    mu_off = float(np.average(off_rank["epa_per_play"], weights=off_rank["plays"]))
    mu_def = float(np.average(def_rank["epa_allowed"],  weights=def_rank["plays_against"]))

    off_rank["epa_per_play_sh"] = (
        (off_rank["epa_per_play"] * off_rank["plays"] + mu_off * N0) /
        (off_rank["plays"] + N0)
    )
    def_rank["epa_allowed_sh"] = (
        (def_rank["epa_allowed"] * def_rank["plays_against"] + mu_def * N0) /
        (def_rank["plays_against"] + N0)
    )

    # ---- pace (unfiltered, exclude only kneel/spike)
    raw = pbp_subset.loc[pbp_subset["play_type"].isin(["pass","run"])].copy()
    kneel_r = bool_series(raw, "qb_kneel")
    spike_r = bool_series(raw, "qb_spike")
    raw = raw[~kneel_r & ~spike_r]

    off_pg = (raw.groupby(["game_id","posteam"]).size().reset_index(name="plays"))
    off_pg = (off_pg.groupby("posteam")["plays"].mean()
              .reset_index().rename(columns={"posteam":"team","plays":"off_plays_pg_raw"}))

    def_pg = (raw.groupby(["game_id","defteam"]).size().reset_index(name="plays"))
    def_pg = (def_pg.groupby("defteam")["plays"].mean()
              .reset_index().rename(columns={"defteam":"team","plays":"def_plays_pg_raw"}))

    off_rank = off_rank.merge(off_pg, on="team", how="left")
    def_rank = def_rank.merge(def_pg, on="team", how="left")

    # ---- Eckel (drive-quality)
    cols = ["game_id","drive","posteam","defteam","first_down","yardline_100",
            "touchdown","rush_touchdown","pass_touchdown"]
    rd = raw[cols].copy()
    rd["is_TD"] = rd[["touchdown","rush_touchdown","pass_touchdown"]].any(axis=1).fillna(False)
    rd["fd_inside40"] = (rd["first_down"].fillna(0).astype(int).eq(1) & (rd["yardline_100"] <= 40))

    drive_flags = (rd.groupby(["game_id","drive","posteam","defteam"], observed=True)
                     .agg(td=("is_TD","any"), fd40=("fd_inside40","any"))
                     .reset_index())
    drive_flags["quality_off"] = drive_flags["td"] | drive_flags["fd40"]

    off_eckel = (drive_flags.groupby("posteam", observed=True)["quality_off"]
                 .agg(drives="size", quality="sum")
                 .assign(off_eckel_rate=lambda x: x["quality"]/x["drives"])
                 .reset_index().rename(columns={"posteam":"team"}))
    def_eckel = (drive_flags.groupby("defteam", observed=True)["quality_off"]
                 .agg(drives_faced="size", quality_allowed="sum")
                 .assign(def_eckel_allowed=lambda x: x["quality_allowed"]/x["drives_faced"])
                 .reset_index().rename(columns={"defteam":"team"}))

    off_rank = off_rank.merge(off_eckel[["team","off_eckel_rate"]], on="team", how="left")
    def_rank = def_rank.merge(def_eckel[["team","def_eckel_allowed"]], on="team", how="left")

    # fill any missing Eckel with league mean from subset
    mu_off_eckel = float(off_rank["off_eckel_rate"].mean(skipna=True)) if off_rank["off_eckel_rate"].notna().any() else 0.35
    mu_def_eckel = float(def_rank["def_eckel_allowed"].mean(skipna=True)) if def_rank["def_eckel_allowed"].notna().any() else 0.35
    off_rank["off_eckel_rate"] = off_rank["off_eckel_rate"].fillna(mu_off_eckel)
    def_rank["def_eckel_allowed"] = def_rank["def_eckel_allowed"].fillna(mu_def_eckel)

    return off_rank, def_rank, mu_off, mu_def

def matchup_frame(sched_wk, off_rank, def_rank, mu_off, mu_def):
    home_off = off_rank[["team","epa_per_play_sh","off_plays_pg_raw","off_eckel_rate"]].rename(
        columns={"team":"home_team","epa_per_play_sh":"home_off_epa","off_plays_pg_raw":"home_off_plays_pg","off_eckel_rate":"home_off_eckel"}
    )
    away_off = off_rank[["team","epa_per_play_sh","off_plays_pg_raw","off_eckel_rate"]].rename(
        columns={"team":"away_team","epa_per_play_sh":"away_off_epa","off_plays_pg_raw":"away_off_plays_pg","off_eckel_rate":"away_off_eckel"}
    )
    home_def = def_rank[["team","epa_allowed_sh","def_plays_pg_raw","def_eckel_allowed"]].rename(
        columns={"team":"home_team","epa_allowed_sh":"home_def_epa_allowed","def_plays_pg_raw":"home_def_plays_pg","def_eckel_allowed":"home_def_eckel_allowed"}
    )
    away_def = def_rank[["team","epa_allowed_sh","def_plays_pg_raw","def_eckel_allowed"]].rename(
        columns={"team":"away_team","epa_allowed_sh":"away_def_epa_allowed","def_plays_pg_raw":"away_def_plays_pg","def_eckel_allowed":"away_def_eckel_allowed"}
    )
    m = (sched_wk
         .merge(home_off, on="home_team", how="left")
         .merge(away_off, on="away_team", how="left")
         .merge(home_def, on="home_team", how="left")
         .merge(away_def, on="away_team", how="left"))

    for c in ["home_off_epa","away_off_epa"]:
        m[c] = m[c].fillna(mu_off)
    for c in ["home_def_epa_allowed","away_def_epa_allowed"]:
        m[c] = m[c].fillna(mu_def)

    # K_pair from offense pace
    plays_pg_all = pd.concat([off_rank["off_plays_pg_raw"], def_rank["def_plays_pg_raw"]], ignore_index=True).dropna()
    if len(plays_pg_all) >= 20:
        K_LO_raw, K_HI_raw = np.percentile(plays_pg_all, [10, 90])
        K_LO = float(max(50.0, K_LO_raw))
        K_HI = float(max(K_LO + 2.0, K_HI_raw))
    else:
        K_LO, K_HI = 55.0, 72.0
    m["K_pair"] = ((m["home_off_plays_pg"].fillna(60) + m["away_off_plays_pg"].fillna(60)) / 2.0).clip(K_LO, K_HI)

    # net EPA (sign-correct), with soft clip scale learned from matchups this week
    tmp = (sched_wk
           .merge(home_off[["home_team","home_off_epa"]], on="home_team", how="left")
           .merge(away_off[["away_team","away_off_epa"]], on="away_team", how="left")
           .merge(home_def[["home_team","home_def_epa_allowed"]], on="home_team", how="left")
           .merge(away_def[["away_team","away_def_epa_allowed"]], on="away_team", how="left"))
    for c in ["home_off_epa","away_off_epa"]:
        tmp[c] = tmp[c].fillna(mu_off)
    for c in ["home_def_epa_allowed","away_def_epa_allowed"]:
        tmp[c] = tmp[c].fillna(mu_def)
    net_unclipped = (tmp["home_off_epa"] + tmp["away_def_epa_allowed"]) - (tmp["away_off_epa"] + tmp["home_def_epa_allowed"])
    net_unclipped = pd.to_numeric(net_unclipped, errors="coerce").dropna()
    scale = float(np.clip(net_unclipped.abs().quantile(0.95), 0.08, 0.20)) if len(net_unclipped) >= 8 else 0.15

    m["home_off_vs_away_def"] = m["home_off_epa"] + m["away_def_epa_allowed"]
    m["away_off_vs_home_def"] = m["away_off_epa"] + m["home_def_epa_allowed"]
    m["net_epa_per_play"] = soft_clip(m["home_off_vs_away_def"] - m["away_off_vs_home_def"], scale)

    return m

# ================== TRAIN: learn weights (EPA & Eckel) ==================
try:
    train_seasons = list(range(SEASON - TRAIN_YEARS_BACK, SEASON))
    pbp_train = import_pbp_data(train_seasons)
    sched_train = import_schedules(train_seasons)

    rows = []
    for s in train_seasons:
        sched_s = sched_train[(sched_train["season"] == s) & (sched_train["game_type"] == "REG")].copy()
        # completed games only
        played_weeks = sorted(sched_s.loc[sched_s["home_score"].notna() & sched_s["away_score"].notna(), "week"].unique())
        if not played_weeks:
            continue

        pbp_s = pbp_train[pbp_train["season"] == s].copy()

        for wk in played_weeks:
            # build features from weeks < wk (no leakage)
            pbp_pre = pbp_s[pbp_s["week"] < wk].copy()
            if pbp_pre.empty:
                continue
            built = build_team_ratings(pbp_pre)
            if built is None:
                continue
            off_r, def_r, mu_off_s, mu_def_s = built

            sched_wk = sched_s[(sched_s["week"] == wk)].copy()
            # only keep rows with final scores
            sched_wk = sched_wk[sched_wk["home_score"].notna() & sched_wk["away_score"].notna()].copy()
            if sched_wk.empty:
                continue

            m_wk = matchup_frame(sched_wk, off_r, def_r, mu_off_s, mu_def_s)

            # DRIVES_PG for the week (using pbp_pre)
            drives_off = (pbp_pre.groupby(["game_id","posteam"])["drive"].nunique().reset_index(name="drives"))
            DRIVES_PG = float(drives_off["drives"].mean()) if len(drives_off) else 11.5
            DRIVES_PG = float(np.clip(DRIVES_PG, 9.0, 13.5))

            # features
            H = (~m_wk.get("neutral_site", pd.Series(False, index=m_wk.index)).fillna(False)).astype(int)  # 1 if not neutral
            X1 = m_wk["K_pair"] * m_wk["net_epa_per_play"]                 # EPA term in points
            # Eckel term in "expected drives worth" units; let regression learn points/drive
            net_eckel = ( (m_wk["home_off_eckel"] - m_wk["away_def_eckel_allowed"])
                        - (m_wk["away_off_eckel"] - m_wk["home_def_eckel_allowed"]) )
            X2 = DRIVES_PG * net_eckel

            y = m_wk["home_score"].astype(float) - m_wk["away_score"].astype(float)

            rows.append(pd.DataFrame({
                "H": H.values,
                "X1": X1.values,
                "X2": X2.values,
                "y": y.values
            }))

    df_hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["H","X1","X2","y"])

    if len(df_hist) >= 100:
        # Linear regression without intercept: y = b1*H + b2*X1 + b3*X2
        X = df_hist[["H","X1","X2"]].to_numpy(dtype=float)
        y = df_hist["y"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        B_HFA, B_EPA, B_ECKEL = map(float, coef)
        resid = y - X.dot(coef)
        SIGMA = float(np.std(resid, ddof=1))
        SIGMA = float(np.clip(SIGMA, 10.0, 17.0))
    else:
        # Fallback if not enough data
        B_HFA, B_EPA, B_ECKEL = 1.3, 1.0, 0.0
        SIGMA = 13.5

except Exception:
    B_HFA, B_EPA, B_ECKEL = 1.3, 1.0, 0.0
    SIGMA = 13.5

# ================== PREDICT THIS WEEK ==================

# ---------- Load current-season PBP once ----------
pbp_curr = import_pbp_data([SEASON])
kneel = bool_series(pbp_curr, "qb_kneel")
spike = bool_series(pbp_curr, "qb_spike")

# filtered snaps for current ratings
df = pbp_curr.loc[
    (pbp_curr["season"] == SEASON)
    & (pbp_curr["play_type"].isin(["pass","run"]))
    & (pbp_curr["epa"].notna())
    & (~kneel) & (~spike)
    & (pbp_curr["wp"].between(WP_MIN, WP_MAX, inclusive="both"))
].copy()

off_rank = (df.groupby("posteam", observed=True)
              .agg(plays=("epa","size"), epa_per_play=("epa","mean"))
              .reset_index().rename(columns={"posteam":"team"}))
def_rank = (df.groupby("defteam", observed=True)
              .agg(plays_against=("epa","size"), epa_allowed=("epa","mean"))
              .reset_index().rename(columns={"defteam":"team"}))
if off_rank.empty or def_rank.empty:
    print("No qualifying 2025 plays found yet. Try after some games are played.")
    raise SystemExit(0)

mu_off = float(np.average(off_rank["epa_per_play"], weights=off_rank["plays"]))
mu_def = float(np.average(def_rank["epa_allowed"],  weights=def_rank["plays_against"]))
off_rank["epa_per_play_sh"] = (
    (off_rank["epa_per_play"] * off_rank["plays"] + mu_off * N0) /
    (off_rank["plays"] + N0)
)
def_rank["epa_allowed_sh"] = (
    (def_rank["epa_allowed"] * def_rank["plays_against"] + mu_def * N0) /
    (def_rank["plays_against"] + N0)
)

# Schedule this week
sched = import_schedules([SEASON])
games = sched.loc[(sched["season"]==SEASON) & (sched["week"]==WEEK) & (sched["game_type"]=="REG")].copy()

# PER-GAME PACE & Eckel from weeks < WEEK
if WEEK > 1:
    recent = pbp_curr[(pbp_curr["season"]==SEASON) & (pbp_curr["week"] < WEEK)].copy()
    kneel_r = bool_series(recent, "qb_kneel")
    spike_r = bool_series(recent, "qb_spike")
    recent = recent[~kneel_r & ~spike_r]

    off_pg = (recent[recent["play_type"].isin(["pass","run"])]
                  .groupby(["game_id","posteam"]).size().reset_index(name="plays"))
    off_pg = (off_pg.groupby("posteam")["plays"].mean()
              .reset_index().rename(columns={"posteam":"team","plays":"off_plays_pg_raw"}))
    def_pg = (recent[recent["play_type"].isin(["pass","run"])]
                  .groupby(["game_id","defteam"]).size().reset_index(name="plays"))
    def_pg = (def_pg.groupby("defteam")["plays"].mean()
              .reset_index().rename(columns={"defteam":"team","plays":"def_plays_pg_raw"}))

    # Eckel
    cols = ["game_id","drive","posteam","defteam","first_down","yardline_100",
            "touchdown","rush_touchdown","pass_touchdown","play_type"]
    rd = recent[cols].copy()
    rd = rd[rd["play_type"].isin(["pass","run"])]
    rd["is_TD"] = rd[["touchdown","rush_touchdown","pass_touchdown"]].any(axis=1).fillna(False)
    rd["fd_inside40"] = (rd["first_down"].fillna(0).astype(int).eq(1) & (rd["yardline_100"] <= 40))
    drive_flags = (rd.groupby(["game_id","drive","posteam","defteam"], observed=True)
                     .agg(td=("is_TD","any"), fd40=("fd_inside40","any"))
                     .reset_index())
    drive_flags["quality_off"] = drive_flags["td"] | drive_flags["fd40"]
    off_eckel = (drive_flags.groupby("posteam", observed=True)["quality_off"]
                 .agg(drives="size", quality="sum")
                 .assign(off_eckel_rate=lambda x: x["quality"]/x["drives"])
                 .reset_index().rename(columns={"posteam":"team"}))
    def_eckel = (drive_flags.groupby("defteam", observed=True)["quality_off"]
                 .agg(drives_faced="size", quality_allowed="sum")
                 .assign(def_eckel_allowed=lambda x: x["quality_allowed"]/x["drives_faced"])
                 .reset_index().rename(columns={"defteam":"team"}))
else:
    off_pg = pd.DataFrame({"team": off_rank["team"], "off_plays_pg_raw": 60.0})
    def_pg = pd.DataFrame({"team": def_rank["team"], "def_plays_pg_raw": 60.0})
    off_eckel = pd.DataFrame({"team": off_rank["team"], "off_eckel_rate": 0.35})
    def_eckel = pd.DataFrame({"team": def_rank["team"], "def_eckel_allowed": 0.35})

# attach to rankings
off_rank = (off_rank.merge(off_pg, on="team", how="left")
                    .merge(off_eckel, on="team", how="left"))
def_rank = (def_rank.merge(def_pg, on="team", how="left")
                    .merge(def_eckel, on="team", how="left"))

# Build matchup features
m = matchup_frame(games, off_rank, def_rank, mu_off, mu_def)

# DRIVES_PG for current predictions
if WEEK > 1 and 'recent' in locals() and not recent.empty:
    drives_off = (recent.groupby(["game_id","posteam"])["drive"]
                        .nunique().reset_index(name="drives"))
    DRIVES_PG = float(drives_off["drives"].mean()) if len(drives_off) else 11.5
else:
    DRIVES_PG = 11.5
DRIVES_PG = float(np.clip(DRIVES_PG, 9.0, 13.5))

# Learned model:
# margin = B_HFA * H + B_EPA * (K_pair * net_epa_per_play) + B_ECKEL * (DRIVES_PG * net_eckel_rate)
H_game = (~m.get("neutral_site", pd.Series(False, index=m.index)).fillna(False)).astype(int)
net_eckel_now = (
    (m["home_off_eckel"] - m["away_def_eckel_allowed"]) -
    (m["away_off_eckel"] - m["home_def_eckel_allowed"])
)

print(f"Learned weights — HFA: {B_HFA:.2f}  |  EPA coeff: {B_EPA:.3f}  |  Eckel coeff: {B_ECKEL:.3f}  |  SIGMA: {SIGMA:.2f}")

m["pred_margin"] = (
    B_HFA * H_game
    + B_EPA   * (m["K_pair"] * m["net_epa_per_play"])
    + B_ECKEL * (DRIVES_PG * net_eckel_now)
)

# Win probability (Normal CDF via erf)
z = m["pred_margin"] / (SIGMA * np.sqrt(2.0))
m["home_win_prob"] = 0.5 * (1.0 + z.apply(math.erf))

m["pick"] = np.where(m["home_win_prob"] >= 0.5, m["home_team"], m["away_team"])
m["pick_prob"] = np.where(m["home_win_prob"] >= 0.5, m["home_win_prob"], 1.0 - m["home_win_prob"])

# ---------- Output ----------
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", "{:.3f}".format)

view = (m[[
    "game_id","week","away_team","home_team",
    "home_off_epa","away_def_epa_allowed","home_off_vs_away_def",
    "away_off_epa","home_def_epa_allowed","away_off_vs_home_def",
    "net_epa_per_play","K_pair","pred_margin","home_win_prob","pick","pick_prob"
]]
    .sort_values(by=["pick_prob","pred_margin"], ascending=[False, False])
    .reset_index(drop=True)
)

print(f"\n=== {SEASON} Week {WEEK} — EPA+Eckel Matchup Model (learned weights) ===\n")
print(view.to_string(index=False))
