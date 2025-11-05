# ================== public API ==================
def predict_week(
    season: int,
    week: int,
    *,
    wp_min: float = 0.07,
    wp_max: float = 0.93,
    n0: int = 175,
    train_years_back: int = 3,
) -> pd.DataFrame:
    """
    Compute model predictions for (season, week) and return a tidy DataFrame ready for JSON serialization.

    Returns columns:
      ['game_id','week','away_team','home_team','home_off_epa','away_def_epa_allowed',
       'home_off_vs_away_def','away_off_epa','home_def_epa_allowed','away_off_vs_home_def',
       'net_epa_per_play','K_pair','pred_margin','home_win_prob','pick','pick_prob']
    """
    # ================== TRAIN: learn weights (EPA & Eckel) ==================
    try:
        train_seasons = list(range(season - train_years_back, season))
        pbp_train = import_pbp_data(train_seasons)
        sched_train = import_schedules(train_seasons)

        rows: list[pd.DataFrame] = []
        for s in train_seasons:
            sched_s = sched_train[(sched_train["season"] == s) & (sched_train["game_type"] == "REG")].copy()
            played_weeks = sorted(
                sched_s.loc[
                    sched_s["home_score"].notna() & sched_s["away_score"].notna(), "week"
                ].unique()
            )
            if not played_weeks:
                continue

            pbp_s = pbp_train[pbp_train["season"] == s].copy()

            for wk in played_weeks:
                # build features from weeks < wk (no leakage)
                pbp_pre = pbp_s[pbp_s["week"] < wk].copy()
                if pbp_pre.empty:
                    continue
                built = build_team_ratings(pbp_pre, wp_min=wp_min, wp_max=wp_max, n0=n0)
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
                drives_off = (
                    pbp_pre.groupby(["game_id", "posteam"])["drive"].nunique().reset_index(name="drives")
                )
                DRIVES_PG = float(drives_off["drives"].mean()) if len(drives_off) else 11.5
                DRIVES_PG = float(np.clip(DRIVES_PG, 9.0, 13.5))

                # features
                H = (
                    ~m_wk.get("neutral_site", pd.Series(False, index=m_wk.index))
                    .fillna(False)
                    .astype(int)
                )  # 1 if not neutral
                X1 = m_wk["K_pair"] * m_wk["net_epa_per_play"]  # EPA term in points
                # Eckel term in "expected drives worth" units; let regression learn points/drive
                net_eckel = (m_wk["home_off_eckel"] - m_wk["away_def_eckel_allowed"]) - (
                    m_wk["away_off_eckel"] - m_wk["home_def_eckel_allowed"]
                )
                X2 = DRIVES_PG * net_eckel

                y = m_wk["home_score"].astype(float) - m_wk["away_score"].astype(float)

                rows.append(pd.DataFrame({"H": H.values, "X1": X1.values, "X2": X2.values, "y": y.values}))

        df_hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["H", "X1", "X2", "y"])

        if len(df_hist) >= 100:
            # Linear regression without intercept: y = b1*H + b2*X1 + b3*X2
            X = df_hist[["H", "X1", "X2"]].to_numpy(dtype=float)
            y_arr = df_hist["y"].to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
            B_HFA, B_EPA, B_ECKEL = map(float, coef)
            resid = y_arr - X.dot(coef)
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
    # Load current-season PBP once
    pbp_curr = import_pbp_data([season])
    kneel = bool_series(pbp_curr, "qb_kneel")
    spike = bool_series(pbp_curr, "qb_spike")

    df = pbp_curr.loc[
        (pbp_curr["season"] == season)
        & (pbp_curr["play_type"].isin(["pass", "run"]))
        & (pbp_curr["epa"].notna())
        & (~kneel)
        & (~spike)
        & (pbp_curr["wp"].between(wp_min, wp_max, inclusive="both"))
    ].copy()

    off_rank = (
        df.groupby("posteam", observed=True)
        .agg(plays=("epa", "size"), epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    def_rank = (
        df.groupby("defteam", observed=True)
        .agg(plays_against=("epa", "size"), epa_allowed=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    if off_rank.empty or def_rank.empty:
        # No qualifying plays yet — return empty frame with expected columns
        return pd.DataFrame(
            columns=[
                "game_id",
                "week",
                "away_team",
                "home_team",
                "home_off_epa",
                "away_def_epa_allowed",
                "home_off_vs_away_def",
                "away_off_epa",
                "home_def_epa_allowed",
                "away_off_vs_home_def",
                "net_epa_per_play",
                "K_pair",
                "pred_margin",
                "home_win_prob",
                "pick",
                "pick_prob",
            ]
        )

    mu_off = float(np.average(off_rank["epa_per_play"], weights=off_rank["plays"]))
    mu_def = float(np.average(def_rank["epa_allowed"], weights=def_rank["plays_against"]))
    off_rank["epa_per_play_sh"] = (off_rank["epa_per_play"] * off_rank["plays"] + mu_off * n0) / (
        off_rank["plays"] + n0
    )
    def_rank["epa_allowed_sh"] = (
        (def_rank["epa_allowed"] * def_rank["plays_against"] + mu_def * n0) / (def_rank["plays_against"] + n0)
    )

    # Schedule this week
    sched = import_schedules([season])
    games = sched.loc[
        (sched["season"] == season) & (sched["week"] == week) & (sched["game_type"] == "REG")
    ].copy()

    # PER-GAME PACE & Eckel from weeks < week
    if week > 1:
        recent = pbp_curr[(pbp_curr["season"] == season) & (pbp_curr["week"] < week)].copy()
        kneel_r = bool_series(recent, "qb_kneel")
        spike_r = bool_series(recent, "qb_spike")
        recent = recent[~kneel_r & ~spike_r]

        off_pg = (
            recent[recent["play_type"].isin(["pass", "run"])]
            .groupby(["game_id", "posteam"])
            .size()
            .reset_index(name="plays")
        )
        off_pg = (
            off_pg.groupby("posteam")["plays"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "plays": "off_plays_pg_raw"})
        )
        def_pg = (
            recent[recent["play_type"].isin(["pass", "run"])]
            .groupby(["game_id", "defteam"])
            .size()
            .reset_index(name="plays")
        )
        def_pg = (
            def_pg.groupby("defteam")["plays"]
            .mean()
            .reset_index()
            .rename(columns={"defteam": "team", "plays": "def_plays_pg_raw"})
        )

        # Eckel
        cols = [
            "game_id",
            "drive",
            "posteam",
            "defteam",
            "first_down",
            "yardline_100",
            "touchdown",
            "rush_touchdown",
            "pass_touchdown",
            "play_type",
        ]
        rd = recent[cols].copy()
        rd = rd[rd["play_type"].isin(["pass", "run"])]
        rd["is_TD"] = rd[["touchdown", "rush_touchdown", "pass_touchdown"]].any(axis=1).fillna(False)
        rd["fd_inside40"] = (rd["first_down"].fillna(0).astype(int).eq(1) & (rd["yardline_100"] <= 40))
        drive_flags = (
            rd.groupby(["game_id", "drive", "posteam", "defteam"], observed=True)
            .agg(td=("is_TD", "any"), fd40=("fd_inside40", "any"))
            .reset_index()
        )
        drive_flags["quality_off"] = drive_flags["td"] | drive_flags["fd40"]
        off_eckel = (
            drive_flags.groupby("posteam", observed=True)["quality_off"]
            .agg(drives="size", quality="sum")
            .assign(off_eckel_rate=lambda x: x["quality"] / x["drives"])
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        def_eckel = (
            drive_flags.groupby("defteam", observed=True)["quality_off"]
            .agg(drives_faced="size", quality_allowed="sum")
            .assign(def_eckel_allowed=lambda x: x["quality_allowed"] / x["drives_faced"])
            .reset_index()
            .rename(columns={"defteam": "team"})
        )
    else:
        off_pg = pd.DataFrame({"team": off_rank["team"], "off_plays_pg_raw": 60.0})
        def_pg = pd.DataFrame({"team": def_rank["team"], "def_plays_pg_raw": 60.0})
        off_eckel = pd.DataFrame({"team": off_rank["team"], "off_eckel_rate": 0.35})
        def_eckel = pd.DataFrame({"team": def_rank["team"], "def_eckel_allowed": 0.35})

    # attach to rankings
    off_rank = off_rank.merge(off_pg, on="team", how="left").merge(off_eckel, on="team", how="left")
    def_rank = def_rank.merge(def_pg, on="team", how="left").merge(def_eckel, on="team", how="left")

    # Build matchup features
    m = matchup_frame(games, off_rank, def_rank, mu_off, mu_def)

    # DRIVES_PG for current predictions
    if week > 1 and "recent" in locals() and not recent.empty:
        drives_off = recent.groupby(["game_id", "posteam"])["drive"].nunique().reset_index(name="drives")
        DRIVES_PG = float(drives_off["drives"].mean()) if len(drives_off) else 11.5
    else:
        DRIVES_PG = 11.5
    DRIVES_PG = float(np.clip(DRIVES_PG, 9.0, 13.5))

    # Learned model:
    # margin = B_HFA * H + B_EPA * (K_pair * net_epa_per_play) + B_ECKEL * (DRIVES_PG * net_eckel_rate)
    H_game = (
        ~m.get("neutral_site", pd.Series(False, index=m.index))
        .fillna(False)
        .astype(int)
    )
    net_eckel_now = (m["home_off_eckel"] - m["away_def_eckel_allowed"]) - (
        m["away_off_eckel"] - m["home_def_eckel_allowed"]
    )

    m["pred_margin"] = (
        B_HFA * H_game
        + B_EPA * (m["K_pair"] * m["net_epa_per_play"])
        + B_ECKEL * (DRIVES_PG * net_eckel_now)
    )

    # Win probability (Normal CDF via erf)
    z = m["pred_margin"] / (SIGMA * np.sqrt(2.0))
    m["home_win_prob"] = 0.5 * (1.0 + z.apply(math.erf))

    m["pick"] = np.where(m["home_win_prob"] >= 0.5, m["home_team"], m["away_team"])
    m["pick_prob"] = np.where(m["home_win_prob"] >= 0.5, m["home_win_prob"], 1.0 - m["home_win_prob"])

    view = (
        m[
            [
                "game_id",
                "week",
                "away_team",
                "home_team",
                "home_off_epa",
                "away_def_epa_allowed",
                "home_off_vs_away_def",
                "away_off_epa",
                "home_def_epa_allowed",
                "away_off_vs_home_def",
                "net_epa_per_play",
                "K_pair",
                "pred_margin",
                "home_win_prob",
                "pick",
                "pick_prob",
            ]
        ]
        .sort_values(by=["pick_prob", "pred_margin"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return view
    
#if __name__ == "__main__":
#    # Quick test
#        df = predict_week(2025, 8)
#        pd.set_option("display.max_rows", None)
#        pd.set_option("display.width", 180)
#        pd.set_option("display.float_format", "{:.3f}".format)
#        print(df.to_string(index=False))   
