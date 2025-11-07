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

  const onSnapshot = async () => {
    try {
      const res = await fetch(`${apiBase}/predict/snapshot?season=${season}&week=${week}`, { method: "POST" });
      if (!res.ok) throw new Error(`Snapshot failed (HTTP ${res.status})`);
      await fetchHistory();
    } catch (e: any) {
      alert(e?.message ?? "Snapshot failed");
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

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="card">
        <div className="card-header">
          <h1 className="text-lg font-semibold">Prediction History</h1>
          <div className="flex items-center gap-2">
            <button onClick={onSnapshot} className="btn btn-secondary">Save snapshot</button>
            <button onClick={fetchHistory} className="btn btn-primary" disabled={loading}>
              {loading ? "Loading…" : "Load"}
            </button>
          </div>
        </div>
        <div className="card-body flex flex-wrap gap-3">
          <label className="text-sm">
            <div className="text-gray-500 mb-1">Season</div>
            <input type="number" className="input w-32" value={season} onChange={(e) => setSeason(parseInt(e.target.value || "0"))} />
          </label>
          <label className="text-sm">
            <div className="text-gray-500 mb-1">Week</div>
            <input type="number" className="input w-24" value={week} onChange={(e) => setWeek(parseInt(e.target.value || "0"))} />
          </label>
          <label className="text-sm">
            <div className="text-gray-500 mb-1">Limit</div>
            <input type="number" className="input w-24" value={limit} onChange={(e) => setLimit(parseInt(e.target.value || "0"))} />
          </label>
          <div className="flex-1" />
          <div className="hidden sm:flex items-center gap-3">
            <div className="text-xs text-gray-500">
              API: <code className="text-gray-700">{apiBase}/history?season={season}&week={week}&limit={limit}</code>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="card-body p-0">
          <table className="table">
            <thead className="thead">
              <tr className="tr">
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
            <tbody>
              {loading && <RowMessage text="Loading…" />}
              {!loading && err && <RowMessage text={`Error: ${err}`} isError />}
              {!loading && !err && sorted.length === 0 && <RowMessage text="No history found." />}

              {!loading && !err && sorted.map((r) => (
                <tr key={r.id + "-" + r.game_id} className="tr">
                  <td className="td whitespace-nowrap">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="td">{r.season}</td>
                  <td className="td">{r.week}</td>
                  <td className="td font-mono text-xs text-gray-600">{r.game_id}</td>
                  <td className="td whitespace-nowrap">
                    <span className="font-medium">{r.away_team}</span> @ <span className="font-medium">{r.home_team}</span>
                  </td>
                  <td className="td text-right num">{fmtNum(r.pred_margin)}</td>
                  <td className="td text-right num">{fmtPct(r.home_win_prob)}</td>
                  <td className="td text-right num">
                    <span className="font-semibold">{r.pick}</span>{" "}
                    <span className="text-gray-600">({fmtPct(r.pick_prob)})</span>
                  </td>
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
      <td className="td text-center py-8 text-sm" colSpan={8}>
        <span className={`px-3 py-2 rounded ${isError ? "bg-red-50 text-red-700 border border-red-200" : "bg-gray-100 text-gray-700"}`}>
          {text}
        </span>
      </td>
    </tr>
  );
}
