from nfl_data_py import import_schedules

SEASON = 2025
sched = import_schedules([SEASON])

wc = sched[(sched["season"] == SEASON) & (sched["game_type"] == "WC")].copy()

print("WC rows:", len(wc))
print("WC weeks:", sorted(wc["week"].dropna().unique().tolist()))

cols = ["season","week","game_type","away_team","home_team","gameday","game_id"]
print(wc[cols].sort_values(["week","gameday"]).head(20).to_string(index=False))