"use client";

import { useEffect, useMemo, useState } from "react";

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
  const [season, setSeason] = useState<number>(2025);
  const [week, setWeek] = useState<number>(8);
  const [rows, setRows] = useState<GameView[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<keyof GameView>("pick_prob");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

  const fetchData = async () => {
    setLoading(true);
    setErr(null);
    try {
      const url = `${apiBase}/predict?season=${season}&week=${week}`;
      const res = await fetch(url, { cache: "no-store" });
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
      // tie-breaker
      const ap = a.pick_prob ?? 0;
      const bp = b.pick_prob ?? 0;
      return bp - ap;
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
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto p-6 space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">NFL Model — Picks & Probabilities</h1>
          <div className="flex gap-2">
            <input
              type="number"
              className="border rounded px-3 py-2 w-28"
              value={season}
              onChange={(e) => setSeason(parseInt(e.target.value || "0"))}
              placeholder="Season"
            />
            <input
              type="number"
              className="border rounded px-3 py-2 w-20"
              value={week}
              onChange={(e) => setWeek(parseInt(e.target.value || "0"))}
              placeholder="Week"
            />
            <button
              onClick={fetchData}
              className="px-4 py-2 rounded bg-black text-white disabled:opacity-60"
              disabled={loading}
            >
              {loading ? "Loading…" : "Load"}
            </button>
          </div>
        </header>

        {err && (
          <div className="p-3 bg-red-100 border border-red-300 text-red-700 rounded">
            {err}
          </div>
        )}

        <div className="overflow-auto bg-white rounded-lg shadow">
          <table className="min-w-full">
            <thead className="bg-gray-100 text-sm">
              <tr>
                <Th label="Matchup" />
                <Th label="Pred Margin" onClick={() => toggleSort("pred_margin")} active={sortKey === "pred_margin"} dir={sortDir} />
                <Th label="Home Win %" onClick={() => toggleSort("home_win_prob")} active={sortKey === "home_win_prob"} dir={sortDir} />
                <Th label="Pick (Prob)" onClick={() => toggleSort("pick_prob")} active={sortKey === "pick_prob"} dir={sortDir} />
                <Th label="Net EPA/play" onClick={() => toggleSort("net_epa_per_play")} active={sortKey === "net_epa_per_play"} dir={sortDir} />
                <Th label="Pace K_pair" onClick={() => toggleSort("K_pair")} active={sortKey === "K_pair"} dir={sortDir} />
              </tr>
            </thead>
            <tbody className="text-sm">
              {sorted.map((g) => {
                const isHomePick = g.pick === g.home_team;
                const winPct = isHomePick ? g.home_win_prob : 1 - g.home_win_prob;
                return (
                  <tr key={String(g.game_id)} className="border-t">
                    <td className="px-3 py-2 whitespace-nowrap">
                      <div className="font-medium">{g.away_team} @ {g.home_team}</div>
                      <div className="text-gray-500 text-xs">Week {g.week} • Game {g.game_id}</div>
                    </td>
                    <td className="px-3 py-2 text-right">{fmtNum(g.pred_margin)}</td>
                    <td className="px-3 py-2 text-right">{fmtPct(g.home_win_prob)}</td>
                    <td className="px-3 py-2 text-right">
                      <span className="font-medium">{g.pick}</span>{" "}
                      <span className="text-gray-500">({fmtPct(g.pick_prob)})</span>
                    </td>
                    <td className="px-3 py-2 text-right">{fmtNum(g.net_epa_per_play, 3)}</td>
                    <td className="px-3 py-2 text-right">{fmtNum(g.K_pair, 1)}</td>
                  </tr>
                );
              })}

              {!loading && sorted.length === 0 && (
                <tr><td className="px-3 py-6 text-center text-gray-500" colSpan={6}>No games found.</td></tr>
              )}
              {loading && (
                <tr><td className="px-3 py-6 text-center text-gray-500" colSpan={6}>Loading…</td></tr>
              )}
            </tbody>
          </table>
        </div>

        <p className="text-xs text-gray-500">
          API: <code>{apiBase}/predict?season={season}&week={week}</code>
        </p>
      </div>
    </main>
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
      className={`px-3 py-2 text-left font-medium ${onClick ? "cursor-pointer select-none" : ""}`}
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
