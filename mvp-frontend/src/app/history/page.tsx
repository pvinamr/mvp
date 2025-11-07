"use client";

import { useEffect, useMemo, useState } from "react";

type HistoryRow = {
  id: number;
  season: number;
  week: number;
  game_id: string;
  created_at: string;
  home_team: string;
  away_team: string;
  pred_margin: number;
  home_win_prob: number;
  pick: string;
  pick_prob: number;
};

const fmtPct = (x: number) => `${(x * 100).toFixed(1)}%`;
const fmtNum = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined ? "—" : x.toFixed(d);

export default function HistoryPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

  const [season, setSeason] = useState<number>(2025);
  const [week, setWeek] = useState<number>(8);
  const [limit, setLimit] = useState<number>(200);
  const [rows, setRows] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<keyof HistoryRow>("created_at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const fetchHistory = async () => {
    setLoading(true);
    setErr(null);
    try {
      const url = `${apiBase}/history?season=${season}&week=${week}&limit=${limit}`;
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: HistoryRow[] = await res.json();
      setRows(data);
    } catch (e: any) {
      setErr(e?.message ?? "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a: any, b: any) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const toggleSort = (key: keyof HistoryRow) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const onSnapshot = async () => {
    try {
      const url = `${apiBase}/predict/snapshot?season=${season}&week=${week}`;
      const res = await fetch(url, { method: "POST" });
      if (!res.ok) throw new Error(`Snapshot failed (HTTP ${res.status})`);
      await fetchHistory();
      alert("Snapshot saved & history refreshed.");
    } catch (e: any) {
      alert(e?.message ?? "Snapshot failed");
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto p-6 space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Prediction History</h1>
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
            <input
              type="number"
              className="border rounded px-3 py-2 w-24"
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value || "0"))}
              placeholder="Limit"
              title="Max rows to return"
            />
            <button
              onClick={fetchHistory}
              className="px-4 py-2 rounded bg-black text-white disabled:opacity-60"
              disabled={loading}
            >
              {loading ? "Loading…" : "Load"}
            </button>
            <button
              onClick={onSnapshot}
              className="px-3 py-2 rounded border bg-white"
              title="Save a new snapshot for this season/week"
            >
              Save snapshot
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
                <Th label="Created At" onClick={() => toggleSort("created_at")} active={sortKey === "created_at"} dir={sortDir} />
                <Th label="Season" onClick={() => toggleSort("season")} active={sortKey === "season"} dir={sortDir} />
                <Th label="Week" onClick={() => toggleSort("week")} active={sortKey === "week"} dir={sortDir} />
                <Th label="Game ID" onClick={() => toggleSort("game_id")} active={sortKey === "game_id"} dir={sortDir} />
                <Th label="Matchup" />
                <Th label="Pred Margin" onClick={() => toggleSort("pred_margin")} active={sortKey === "pred_margin"} dir={sortDir} />
                <Th label="Home Win %" onClick={() => toggleSort("home_win_prob")} active={sortKey === "home_win_prob"} dir={sortDir} />
                <Th label="Pick (Prob)" onClick={() => toggleSort("pick_prob")} active={sortKey === "pick_prob"} dir={sortDir} />
              </tr>
            </thead>
            <tbody className="text-sm">
              {sorted.map((r) => (
                <tr key={r.id + "-" + r.game_id} className="border-t">
                  <td className="px-3 py-2 whitespace-nowrap">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="px-3 py-2">{r.season}</td>
                  <td className="px-3 py-2">{r.week}</td>
                  <td className="px-3 py-2 font-mono text-xs text-gray-600">{r.game_id}</td>
                  <td className="px-3 py-2 whitespace-nowrap">
                    <span className="font-medium">{r.away_team}</span> @ <span className="font-medium">{r.home_team}</span>
                  </td>
                  <td className="px-3 py-2 text-right font-semibold font-mono text-gray-900">{fmtNum(r.pred_margin)}</td>
                  <td className="px-3 py-2 text-right font-semibold font-mono text-gray-900">{fmtPct(r.home_win_prob)}</td>
                  <td className="px-3 py-2 text-right font-semibold font-mono text-gray-900">
                    <span className="font-semibold">{r.pick}</span>{" "}
                    <span className="text-gray-600">({fmtPct(r.pick_prob)})</span>
                  </td>
                </tr>
              ))}

              {!loading && sorted.length === 0 && (
                <tr><td className="px-3 py-6 text-center text-gray-500" colSpan={8}>No history found.</td></tr>
              )}
              {loading && (
                <tr><td className="px-3 py-6 text-center text-gray-500" colSpan={8}>Loading…</td></tr>
              )}
            </tbody>
          </table>
        </div>

        <p className="text-xs text-gray-500">
          API: <code>{apiBase}/history?season={season}&week={week}&limit={limit}</code>
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
