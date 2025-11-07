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
  pred_margin: number;
  home_win_prob: number;
  pick: string;
  pick_prob: number;
};

const fmtPct = (x: number) => `${(x * 100).toFixed(1)}%`;
const fmtNum = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined ? "—" : x.toFixed(d);

export default function Home() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
  const [season, setSeason] = useState<number>(2025);
  const [week, setWeek] = useState<number>(8);
  const [rows, setRows] = useState<GameView[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<keyof GameView>("pick_prob");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const fetchData = async () => {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBase}/predict?season=${season}&week=${week}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: GameView[] = await res.json();
      setRows(data);
    } catch (e: any) {
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
      const av = a[sortKey] ?? -Infinity;
      const bv = b[sortKey] ?? -Infinity;
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return (b.pick_prob ?? 0) - (a.pick_prob ?? 0); // tie-breaker
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
            <h1 className="text-lg font-semibold">Picks & Probabilities</h1>
            <span className="badge">Live</span>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/history" className="btn btn-secondary">View History</Link>
            <button onClick={fetchData} className="btn btn-primary" disabled={loading}>
              {loading ? "Loading…" : "Load"}
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
              onChange={(e) => setWeek(parseInt(e.target.value || "0"))}
            />
          </label>
          <div className="flex-1" />
          <div className="hidden sm:flex items-center gap-3">
            <div className="text-xs text-gray-500">
              API: <code className="text-gray-700">{apiBase}/predict?season={season}&week={week}</code>
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
                <Th label="Pred Margin" onClick={() => toggleSort("pred_margin")} active={sortKey === "pred_margin"} dir={sortDir} />
                <Th label="Home Win %" onClick={() => toggleSort("home_win_prob")} active={sortKey === "home_win_prob"} dir={sortDir} />
                <Th label="Pick (Prob)" onClick={() => toggleSort("pick_prob")} active={sortKey === "pick_prob"} dir={sortDir} />
                <Th label="Net EPA/play" onClick={() => toggleSort("net_epa_per_play")} active={sortKey === "net_epa_per_play"} dir={sortDir} />
                <Th label="Pace K_pair" onClick={() => toggleSort("K_pair")} active={sortKey === "K_pair"} dir={sortDir} />
              </tr>
            </thead>
            <tbody>
              {loading && <RowMessage text="Loading…" />}
              {!loading && err && <RowMessage text={`Error: ${err}`} isError />}
              {!loading && !err && sorted.length === 0 && <RowMessage text="No games found." />}

              {!loading && !err && sorted.map((g) => (
                <tr key={String(g.game_id)} className="tr">
                  <td className="td whitespace-nowrap">
                    <div className="font-medium">{g.away_team} @ {g.home_team}</div>
                    <div className="text-gray-500 text-xs">Week {g.week} • Game {g.game_id}</div>
                  </td>
                  <td className="td text-right num">{fmtNum(g.pred_margin)}</td>
                  <td className="td text-right num">{fmtPct(g.home_win_prob)}</td>
                  <td className="td text-right num">
                    <span className="font-semibold">{g.pick}</span>{" "}
                    <span className="text-gray-600">({fmtPct(g.pick_prob)})</span>
                  </td>
                  <td className="td text-right num">{fmtNum(g.net_epa_per_play, 3)}</td>
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
    <th className="th select-none" onClick={onClick} title={onClick ? "Sort" : undefined}>
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
      <td className="td text-center py-8 text-sm" colSpan={6}>
        <span className={`px-3 py-2 rounded ${isError ? "bg-red-50 text-red-700 border border-red-200" : "bg-gray-100 text-gray-700"}`}>
          {text}
        </span>
      </td>
    </tr>
  );
}
