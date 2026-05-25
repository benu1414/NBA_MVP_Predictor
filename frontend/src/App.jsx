import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { BarChart3, FlaskConical, RefreshCw, ShieldCheck, UserRound } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function App() {
  const [seasons, setSeasons] = useState([]);
  const [selectedYear, setSelectedYear] = useState(2021);
  const [predictions, setPredictions] = useState([]);
  const [actuals, setActuals] = useState([]);
  const [models, setModels] = useState(null);
  const [narrative, setNarrative] = useState(null);
  const [qa, setQa] = useState(null);
  const [selectedPlayer, setSelectedPlayer] = useState("");
  const [playerHistory, setPlayerHistory] = useState([]);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [simResult, setSimResult] = useState(null);

  useEffect(() => {
    getJson("/seasons")
      .then((years) => {
        setSeasons(years);
        setSelectedYear(years[years.length - 1] ?? 2021);
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedYear) return;
    setLoading(true);
    Promise.all([
      getJson(`/seasons/${selectedYear}/predictions`),
      getJson(`/seasons/${selectedYear}/actual-results`),
      getJson("/models"),
      getJson("/narrative/status"),
      getJson("/qa"),
    ])
      .then(([predictionRows, actualRows, modelRows, narrativeRows, qaRows]) => {
        setPredictions(predictionRows);
        setActuals(actualRows);
        setModels(modelRows);
        setNarrative(narrativeRows);
        setQa(qaRows);
        setSelectedPlayer(predictionRows[0]?.Player ?? "");
        setError("");
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [selectedYear]);

  useEffect(() => {
    if (!selectedPlayer || !selectedYear) return;
    Promise.all([
      getJson(`/players/${encodeURIComponent(selectedPlayer)}`),
      getJson(`/players/${encodeURIComponent(selectedPlayer)}/seasons/${selectedYear}/explanation`),
    ])
      .then(([historyRows, explanationRows]) => {
        setPlayerHistory(historyRows);
        setExplanation(explanationRows);
      })
      .catch(() => {
        setPlayerHistory([]);
        setExplanation(null);
      });
  }, [selectedPlayer, selectedYear]);

  const chartRows = useMemo(
    () =>
      predictions.slice(0, 10).map((row) => ({
        name: row.Player,
        prediction: Number(row.prediction).toFixed(3),
        actual: Number(row.Share).toFixed(3),
      })),
    [predictions]
  );

  async function runSimulation(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = Object.fromEntries(form.entries());
    for (const key of Object.keys(payload)) {
      if (key !== "player" && payload[key] !== "") {
        payload[key] = Number(payload[key]);
      }
      if (payload[key] === "") {
        delete payload[key];
      }
    }
    payload.year = selectedYear;

    const response = await fetch(`${API_BASE}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setSimResult(await response.json());
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <h1>NBA MVP Predictor</h1>
          <p>Historical rankings, model comparison, and what-if MVP cases.</p>
        </div>
        <label className="season-picker">
          Season
          <select value={selectedYear} onChange={(event) => setSelectedYear(Number(event.target.value))}>
            {seasons.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </label>
      </section>

      {error && <div className="notice">{error}</div>}
      {loading && <div className="notice">Loading season data...</div>}

      <section className="summary-grid">
        <div className="metric">
          <span>Best Model</span>
          <strong>{models?.best_model ?? "train first"}</strong>
        </div>
        <div className="metric">
          <span>Top Prediction</span>
          <strong>{predictions[0]?.Player ?? "none"}</strong>
        </div>
        <div className="metric">
          <span>Actual Winner</span>
          <strong>{actuals[0]?.Player ?? "none"}</strong>
        </div>
        <div className="metric">
          <span>Data QA</span>
          <strong>{qa?.status ?? "unknown"}</strong>
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="panel wide">
          <div className="panel-title">
            <BarChart3 size={18} />
            <h2>Top 10 Model Score vs Actual Share</h2>
          </div>
          <div className="chart-frame">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="prediction" fill="#247BA0" />
                <Bar dataKey="actual" fill="#F25F5C" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">
            <RefreshCw size={18} />
            <h2>Model Metrics</h2>
          </div>
          {models?.best_model && (
            <table className="compact-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Top 5 AP</th>
                  <th>Winner</th>
                  <th>MSE</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(models.models).map(([name, payload]) => (
                  <tr key={name}>
                    <td>{name === models.best_model ? `${name} *` : name}</td>
                    <td>{payload.summary.top_5_average_precision.toFixed(3)}</td>
                    <td>{payload.summary.top_1_accuracy.toFixed(3)}</td>
                    <td>{payload.summary.mse.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="narrative-box">
            <strong>Narrative Layer</strong>
            <span>{narrative?.enabled ? "Enabled" : "Template ready"}</span>
            <small>{narrative?.method}</small>
          </div>
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="panel">
          <div className="panel-title">
            <UserRound size={18} />
            <h2>Player Profile</h2>
          </div>
          <label className="field-label">
            Player
            <select value={selectedPlayer} onChange={(event) => setSelectedPlayer(event.target.value)}>
              {predictions.slice(0, 40).map((row) => (
                <option key={row.Player} value={row.Player}>
                  {row.Player}
                </option>
              ))}
            </select>
          </label>
          {explanation && (
            <div className="profile-grid">
              <div>
                <span className="eyebrow">Predicted Share</span>
                <strong>{Number(explanation.prediction).toFixed(3)}</strong>
              </div>
              <div>
                <span className="eyebrow">Actual Share</span>
                <strong>{Number(explanation.actual_share).toFixed(3)}</strong>
              </div>
              <div>
                <span className="eyebrow">Seasons</span>
                <strong>{playerHistory.length}</strong>
              </div>
              <div>
                <span className="eyebrow">Model</span>
                <strong>{explanation.model}</strong>
              </div>
            </div>
          )}
          <div className="factor-grid">
            <div>
              <h3>Lift</h3>
              {(explanation?.positive_factors ?? []).slice(0, 5).map((factor) => (
                <div className="factor" key={`pos-${factor.feature}`}>
                  <span>{factor.feature}</span>
                  <strong>{factor.contribution.toFixed(3)}</strong>
                </div>
              ))}
            </div>
            <div>
              <h3>Drag</h3>
              {(explanation?.negative_factors ?? []).slice(0, 5).map((factor) => (
                <div className="factor" key={`neg-${factor.feature}`}>
                  <span>{factor.feature}</span>
                  <strong>{factor.contribution.toFixed(3)}</strong>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">
            <ShieldCheck size={18} />
            <h2>Data Quality</h2>
          </div>
          <dl className="metrics-list">
            <dt>Rows</dt>
            <dd>{qa?.source_rows ?? "n/a"}</dd>
            <dt>Years</dt>
            <dd>
              {qa?.year_min ?? "n/a"}-{qa?.year_max ?? "n/a"}
            </dd>
            <dt>Duplicates</dt>
            <dd>{qa?.duplicate_player_years ?? "n/a"}</dd>
            <dt>Missing Teams</dt>
            <dd>{qa?.missing_team_rows ?? "n/a"}</dd>
          </dl>
          <div className="mini-history">
            <h3>Recent Seasons</h3>
            {Object.entries(qa?.row_counts_by_year ?? {})
              .slice(-5)
              .map(([year, count]) => (
                <div className="factor" key={year}>
                  <span>{year}</span>
                  <strong>{count}</strong>
                </div>
              ))}
          </div>
        </div>
      </section>

      <section className="table-grid">
        <div className="panel">
          <h2>Predicted Leaderboard</h2>
          <table>
            <thead>
              <tr>
                <th>Rk</th>
                <th>Player</th>
                <th>Team</th>
                <th>Score</th>
                <th>Actual</th>
              </tr>
            </thead>
            <tbody>
              {predictions.slice(0, 15).map((row) => (
                <tr
                  key={`${row.Player}-${row.Predicted_Rk}`}
                  className={row.Player === selectedPlayer ? "selected-row" : ""}
                  onClick={() => setSelectedPlayer(row.Player)}
                >
                  <td>{row.Predicted_Rk}</td>
                  <td>{row.Player}</td>
                  <td>{row.Tm}</td>
                  <td>{Number(row.prediction).toFixed(3)}</td>
                  <td>{Number(row.Share).toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel">
          <h2>What-If Simulator</h2>
          <form className="sim-form" onSubmit={runSimulation}>
            <input name="player" placeholder="Player name" defaultValue={predictions[0]?.Player ?? ""} />
            <input name="pts" placeholder="PTS" type="number" step="0.1" />
            <input name="ast" placeholder="AST" type="number" step="0.1" />
            <input name="trb" placeholder="REB" type="number" step="0.1" />
            <input name="wins" placeholder="Team wins" type="number" step="1" />
            <input name="games" placeholder="Games" type="number" step="1" />
            <button type="submit">
              <FlaskConical size={16} />
              Simulate
            </button>
          </form>
          {simResult && (
            <div className="simulation-result">
              <strong>{simResult.player}</strong>
              <span>Predicted share: {Number(simResult.predicted_share).toFixed(3)}</span>
              <span>Estimated rank: {simResult.estimated_rank ?? "n/a"}</span>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
