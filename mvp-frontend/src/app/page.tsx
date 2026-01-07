"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

type GameView = {
  game_id: string | number;
  week: number;
  away_team: string;
  home_team: string;
  home_off_epa?: number | null;
  away_def_epa_allowed?: number | null;
  home_off_vs_away_def?: number | null;
  away_off_epa?: number | null;
  home_def_epa_allowed?: number | null;
  away_off_vs_home_def?: number | null;
  net_epa_per_play?: number | null;
  K_pair?: number | null;
  pred_margin: number;        // model home - away
  home_win_prob: number;
  pick: string;
  pick_prob: number;

  // Added fields (from DK odds)
  home_spread?: number | null; // DK spread for home team (e.g., -3.5)
  away_spread?: number | null; // DK spread for away team (optional)
  edge?: number | null;        // model margin vs market implied margin (for home)
};

type SpreadOutcome = {
  team: string;
  line?: number | null;
  price?: number | null;
};

type OddsGame = {
  home_team: string;
  away_team: string;
  commence_time: string;
  spread: SpreadOutcome[];
};

const fmtPct = (x: number) => `${(x * 100).toFixed(1)}%`;
const fmtNum = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined ? "—" : x.toFixed(d);


const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

console.log("API_BASE from env:", apiBase);

// --- helpers to merge odds with model ---

function normalizeName(name: string | undefined | null): string {
  if (!name) return "";
  return name.replace(/[^A-Za-z0-9]/g, "").toUpperCase();
}

// If your model uses abbreviations (e.g. "BUF") and odds uses full names (e.g. "Buffalo Bills"),
// you can add a mapping here:
const TEAM_MAP: Record<string, string> = {
  // NFC
  ARI: "ARIZONACARDINALS",
  ATL: "ATLANTAFALCONS",
  CAR: "CAROLINAPANTHERS",
  CHI: "CHICAGOBEARS",
  DAL: "DALLASCOWBOYS",
  DET: "DETROITLIONS",
  GB:  "GREENBAYPACKERS",
  LA:  "LOSANGELESRAMS",
  MIN: "MINNESOTAVIKINGS",
  NO:  "NEWORLEANSSAINTS",
  NYG: "NEWYORKGIANTS",
  PHI: "PHILADELPHIAEAGLES",
  SEA: "SEATTLESEAHAWKS",
  SF:  "SANFRANCISCO49ERS",
  TB:  "TAMPABAYBUCCANEERS",
  WSH: "WASHINGTONCOMMANDERS",

  // AFC
  BAL: "BALTIMORERAVENS",
  BUF: "BUFFALOBILLS",
  CIN: "CINCINNATIBENGALS",
  CLE: "CLEVELANDBROWNS",
  DEN: "DENVERBRONCOS",
  HOU: "HOUSTONTEXANS",
  IND: "INDIANAPOLISCOLTS",
  JAX: "JACKSONVILLEJAGUARS",
  KC:  "KANSASCITYCHIEFS",
  LAC: "LOSANGELESCHARGERS",
  LV:  "LASVEGASRAIDERS",
  MIA: "MIAMIDOLPHINS",
  NE:  "NEWENGLANDPATRIOTS",
  NYJ: "NEWYORKJETS",
  PIT: "PITTSBURGHSTEELERS",
  TEN: "TENNESSEETITANS",
};


function toComparable(name: string): string {
  const norm = normalizeName(name);
  if (TEAM_MAP[norm]) return TEAM_MAP[norm];
  return norm;
}

function findOddsForGame(game: GameView, odds: OddsGame[]): { homeLine: number | null; awayLine: number | null } {
  const homeKey = toComparable(game.home_team);
  const awayKey = toComparable(game.away_team);

  const match = odds.find((o) => {
    const oHome = toComparable(o.home_team);
    const oAway = toComparable(o.away_team);
    return oHome === homeKey && oAway === awayKey;
  });

  if (!match) {
    return { homeLine: null, awayLine: null };
  }

  let homeOutcome = match.spread.find((s) => toComparable(s.team) === homeKey);
  let awayOutcome = match.spread.find((s) => toComparable(s.team) === awayKey);

  let homeLine = homeOutcome?.line ?? null;
  let awayLine = awayOutcome?.line ?? null;

  // if only one side is present, infer the other
  if (homeLine == null && awayLine != null) {
    homeLine = -awayLine;
  }
  if (awayLine == null && homeLine != null) {
    awayLine = -homeLine;
  }

  return { homeLine, awayLine };
}

// Given model margin (home - away) and home spread (e.g., -3.5),
// market-implied expected margin is roughly -homeSpread.
// Edge = model_margin - market_margin = model_margin + homeSpread.
function computeEdge(predMargin: number, homeSpread: number | null): number | null {
  if (homeSpread == null || Number.isNaN(predMargin)) return null;
  return predMargin + homeSpread;
}

// --------------------------------------------

export default function Home() {
  const [season, setSeason] = useState<number>(2025);
  const [week, setWeek] = useState<number>(18);
  const [rows, setRows] = useState<GameView[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<keyof GameView>("edge");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const fetchData = async () => {
    setLoading(true);
    setErr(null);
    try {
      // fetch predictions & odds in parallel
      const [predRes, oddsRes] = await Promise.all([
        fetch(`${apiBase}/predict?season=${season}&week=${week}`, { cache: "no-store" }),
        fetch(`${apiBase}/odds/draftkings`, { cache: "no-store" }),
      ]);

      if (!predRes.ok) throw new Error(`Predict HTTP ${predRes.status}`);
      const preds: GameView[] = await predRes.json();

      let odds: OddsGame[] = [];
      if (oddsRes.ok) {
        odds = await oddsRes.json();
      }

      // Enrich predictions with DK spread + edge
      const enriched: GameView[] = preds.map((g) => {
        const { homeLine, awayLine } = findOddsForGame(g, odds);
        const edge = computeEdge(g.pred_margin, homeLine);
        return {
          ...g,
          home_spread: homeLine,
          away_spread: awayLine,
          edge,
        };
      });

      setRows(enriched);
    } catch (e: any) {
      console.error(e);
      setErr(e?.message ?? "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a: any, b: any) => {
      const av = a[sortKey];
      const bv = b[sortKey];

      // numeric sort when possible
      if (typeof av === "number" && typeof bv === "number") {
        if (Number.isNaN(av) || av === null) return 1;
        if (Number.isNaN(bv) || bv === null) return -1;
        if (av < bv) return sortDir === "asc" ? -1 : 1;
        if (av > bv) return sortDir === "asc" ? 1 : -1;
        // tie-break by pick_prob
        const ap = (a.pick_prob ?? 0) as number;
        const bp = (b.pick_prob ?? 0) as number;
        return bp - ap;
      }

      // fallback string sort
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const toggleSort = (key: keyof GameView) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  return (
    <div className="space-y-6">
      {/* Controls Card */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">Picks, Market Spreads & Edge</h1>
            <span className="badge">Model vs DraftKings</span>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/history" className="btn btn-secondary">
              View History
            </Link>
            <button onClick={fetchData} className="btn btn-primary" disabled={loading}>
              {loading ? "Loading…" : "Reload"}
            </button>
          </div>
        </div>
        <div className="card-body flex flex-wrap gap-3">
          <label className="text-sm">
            <div className="text-gray-500 mb-1">Season</div>
            <input
              type="number"
              className="input w-32"
              value={season}
              onChange={(e) => setSeason(parseInt(e.target.value || "0"))}
            />
          </label>
          <label className="text-sm">
            <div className="text-gray-500 mb-1">Week</div>
            <input
              type="number"
              className="input w-24"
              value={week}
              min={1}
              max={22}
              onChange={(e) => {
                const value = Number(e.target.value);
                const clamped = Math.min(22, Math.max(1, value));
                setWeek(clamped);
              }}
            />
          </label>
          <div className="flex-1" />
          <div className="hidden sm:flex items-center gap-3">
            <div className="text-xs text-gray-500">
              API:{" "}
              <code className="text-gray-700">
                {apiBase}/predict?season={season}&week={week}
              </code>
            </div>
          </div>
        </div>
      </div>

      {/* Table Card */}
      <div className="card overflow-hidden">
        <div className="card-body p-0">
          <table className="table">
            <thead className="thead">
              <tr className="tr">
                <Th label="Matchup" />
                <Th
                  label="Pred Margin"
                  onClick={() => toggleSort("pred_margin")}
                  active={sortKey === "pred_margin"}
                  dir={sortDir}
                />
                <Th
                  label="Home Win %"
                  onClick={() => toggleSort("home_win_prob")}
                  active={sortKey === "home_win_prob"}
                  dir={sortDir}
                />
                <Th
                  label="Pick (Prob)"
                  onClick={() => toggleSort("pick_prob")}
                  active={sortKey === "pick_prob"}
                  dir={sortDir}
                />
                <Th
                  label="DK Home Spread"
                  onClick={() => toggleSort("home_spread")}
                  active={sortKey === "home_spread"}
                  dir={sortDir}
                />
                <Th
                  label="Edge vs DK (pts)"
                  onClick={() => toggleSort("edge")}
                  active={sortKey === "edge"}
                  dir={sortDir}
                />
                <Th
                  label="Net EPA/play"
                  onClick={() => toggleSort("net_epa_per_play")}
                  active={sortKey === "net_epa_per_play"}
                  dir={sortDir}
                />
                <Th
                  label="Pace K_pair"
                  onClick={() => toggleSort("K_pair")}
                  active={sortKey === "K_pair"}
                  dir={sortDir}
                />
              </tr>
            </thead>
            <tbody>
              {loading && <RowMessage text="Loading…" />}
              {!loading && err && <RowMessage text={`Error: ${err}`} isError />}
              {!loading && !err && sorted.length === 0 && <RowMessage text="No games found." />}

              {!loading &&
                !err &&
                sorted.map((g) => (
                  <tr key={String(g.game_id)} className="tr">
                    <td className="td whitespace-nowrap">
                      <div className="font-medium">
                        {g.away_team} @ {g.home_team}
                      </div>
                      <div className="text-gray-500 text-xs">
                        Week {g.week} • Game {g.game_id}
                      </div>
                    </td>
                    <td className="td text-right num">{fmtNum(g.pred_margin)}</td>
                    <td className="td text-right num">{fmtPct(g.home_win_prob)}</td>
                    <td className="td text-right num">
                      <span className="font-semibold">{g.pick}</span>{" "}
                      <span className="text-gray-600">({fmtPct(g.pick_prob)})</span>
                    </td>
                    <td className="td text-right num">
                      {g.home_spread != null ? g.home_spread.toFixed(1) : "—"}
                    </td>
                    <td className="td text-right num">
                      {g.edge != null ? g.edge.toFixed(1) : "—"}
                    </td>
                    <td className="td text-right num">
                      {fmtNum(g.net_epa_per_play, 3)}
                    </td>
                    <td className="td text-right num">{fmtNum(g.K_pair, 1)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Th({
  label,
  onClick,
  active,
  dir,
}: {
  label: string;
  onClick?: () => void;
  active?: boolean;
  dir?: "asc" | "desc";
}) {
  return (
    <th
      className="th select-none"
      onClick={onClick}
      title={onClick ? "Sort" : undefined}
    >
      <div className="inline-flex items-center gap-1">
        {label}
        {active && <span className="text-gray-400">{dir === "asc" ? "▲" : "▼"}</span>}
      </div>
    </th>
  );
}

function RowMessage({ text, isError = false }: { text: string; isError?: boolean }) {
  return (
    <tr>
      <td className="td text-center py-8 text-sm" colSpan={8}>
        <span
          className={`px-3 py-2 rounded ${
            isError
              ? "bg-red-50 text-red-700 border border-red-200"
              : "bg-gray-100 text-gray-700"
          }`}
        >
          {text}
        </span>
      </td>
    </tr>
  );
}
