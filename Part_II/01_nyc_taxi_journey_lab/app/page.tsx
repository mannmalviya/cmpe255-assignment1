"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

const places = {
  "Grand Central Terminal": [-73.9772, 40.7527], "Times Square": [-73.9855, 40.7580],
  "Central Park South": [-73.9735, 40.7648], "Penn Station": [-73.9935, 40.7506],
  "Wall Street": [-74.0090, 40.7069], SoHo: [-74.0007, 40.7233],
  "Chelsea Market": [-74.0060, 40.7424], "LaGuardia Airport": [-73.8740, 40.7769],
  "JFK Airport": [-73.7781, 40.6413], "Brooklyn Bridge": [-73.9969, 40.7061],
} as const;

type Place = keyof typeof places;
type Prediction = { duration_minutes: number; distance_km: number; estimated_fare_usd: number; traffic_band: string; model_source: string };
type ModelInfo = { status: string; model: string; metrics: Record<string, number | string>; feature_count: number; data_source: string };

function mapPoint([lon, lat]: readonly number[]) {
  const x = 110 + ((lon + 74.15) / .45) * 490;
  const y = 520 - ((lat - 40.55) / .4) * 480;
  return [Math.max(55, Math.min(645, x)), Math.max(30, Math.min(530, y))];
}

function RouteMap({ pickup, dropoff, distance }: { pickup: Place; dropoff: Place; distance?: number }) {
  const a = mapPoint(places[pickup]), b = mapPoint(places[dropoff]);
  const curve = `M${a[0]} ${a[1]} Q${(a[0] + b[0]) / 2 + 35} ${(a[1] + b[1]) / 2} ${b[0]} ${b[1]}`;
  return <div className="map-card">
    <div className="map-toolbar"><span>ROUTE PREVIEW</span><span>{distance ? `${distance.toFixed(1)} KM` : "— KM"}</span></div>
    <svg id="cityMap" viewBox="0 0 700 560" role="img" aria-label="Stylized Manhattan route map">
      <defs><pattern id="grid" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M44 0H0V44" fill="none" stroke="currentColor" strokeWidth="1" /></pattern></defs>
      <path className="island" d="M445 18C419 64 402 107 379 151c-27 53-46 106-72 158-34 68-67 132-84 204l65 31c32-54 64-107 95-162 34-61 57-129 78-196 16-51 42-101 49-154z" />
      <path className="waterline" d="M142 530C236 369 320 183 405 10M302 558C378 408 459 211 530 34" />
      <path className="avenue" d="M267 492 454 58M298 507 475 76M240 471 430 43" />
      <path className="street" d="m260 421 82 40m-65-80 82 40m-64-82 82 40m-65-80 82 40m-65-81 82 39m-65-80 82 39m-64-81 80 39" />
      <rect className="map-grid" width="700" height="560" fill="url(#grid)" />
      <path className="route shadow" d={curve} /><path className="route" d={curve} />
      <circle className="pickup-dot" cx={a[0]} cy={a[1]} r="9" /><circle className="dropoff-dot" cx={b[0]} cy={b[1]} r="9" />
      <text x="470" y="130">UPPER EAST</text><text x="327" y="292">MIDTOWN</text><text x="218" y="492">DOWNTOWN</text>
    </svg>
    <div className="map-legend"><span><i className="pickup-key" />Pickup</span><span><i className="dropoff-key" />Drop-off</span><span>40.7128° N · 74.0060° W</span></div>
  </div>;
}

export default function Home() {
  const [pickup, setPickup] = useState<Place>("Grand Central Terminal");
  const [dropoff, setDropoff] = useState<Place>("Wall Street");
  const [dateTime, setDateTime] = useState("");
  const [passengers, setPassengers] = useState(1);
  const [prediction, setPrediction] = useState<Prediction>();
  const [model, setModel] = useState<ModelInfo>();
  const [online, setOnline] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const placeNames = useMemo(() => Object.keys(places) as Place[], []);

  useEffect(() => {
    // Default to the visitor's local time after mount (avoids SSR hydration mismatch).
    const timer = setTimeout(() => {
      const now = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
      setDateTime(now.toISOString().slice(0, 16));
    });
    fetch("/api/model").then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then((data: ModelInfo) => { setModel(data); setOnline(true); }).catch(() => setOnline(false));
    return () => clearTimeout(timer);
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault(); setLoading(true); setError("");
    const [pickup_longitude, pickup_latitude] = places[pickup];
    const [dropoff_longitude, dropoff_latitude] = places[dropoff];
    try {
      const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
          pickup_datetime: dateTime, passenger_count: passengers, vendor_id: 1 }) });
      if (!response.ok) throw new Error();
      setPrediction(await response.json());
    } catch {
      setError("Could not get an estimate. Check the inputs and try again.");
    } finally { setLoading(false); }
  }

  return <><div className="grain" /><header className="nav shell">
    <a className="brand" href="#top"><span className="brand-mark">CW</span><span>CABWISE</span></a>
    <nav aria-label="Main navigation"><a href="#estimator">Estimator</a><a href="#method">Method</a><a href="#model">Model</a></nav>
    <span className={`status ${online ? "online" : ""}`}><i /><span>{online ? "API ONLINE" : "API OFFLINE"}</span></span>
  </header><main id="top">
    <section className="hero shell"><div className="eyebrow">NYC / MACHINE LEARNING / 2016</div>
      <h1>Know the city<br /><em>before you move.</em></h1>
      <p>Trip-duration intelligence designed for 1.45 million New York taxi journeys. Explore the route, traffic context, and model logic behind every estimate.</p>
      <a className="jump" href="#estimator">Plan a journey <span>↓</span></a><div className="hero-stat"><strong>16</strong><span>engineered<br />signals</span></div>
    </section>
    <section className="estimator shell" id="estimator"><RouteMap pickup={pickup} dropoff={dropoff} distance={prediction?.distance_km} />
      <div className="control-card"><div className="section-tag">01 / JOURNEY INPUT</div><h2>Where are you heading?</h2>
        <form onSubmit={submit}>
          <label>Pickup point<select value={pickup} onChange={(e) => setPickup(e.target.value as Place)}>{placeNames.map((name) => <option key={name}>{name}</option>)}</select></label>
          <label>Drop-off point<select value={dropoff} onChange={(e) => setDropoff(e.target.value as Place)}>{placeNames.map((name) => <option key={name}>{name}</option>)}</select></label>
          <div className="form-row"><label>Date & time<input type="datetime-local" value={dateTime} onChange={(e) => setDateTime(e.target.value)} required /></label>
            <label>Riders<select value={passengers} onChange={(e) => setPassengers(Number(e.target.value))}>{[1,2,3,4,5,6].map((n) => <option key={n}>{n}</option>)}</select></label></div>
          <button type="submit" disabled={loading}><span>{loading ? "Calculating…" : "Estimate this journey"}</span><span>→</span></button>
        </form>
        {pickup === dropoff && <p className="disclaimer">Pickup and drop-off are the same place.</p>}
        {error && <p className="disclaimer" role="alert">{error}</p>}
        <div className="result" aria-live="polite"><div className="result-head"><span>ESTIMATED DURATION</span><span>{prediction ? `${prediction.traffic_band.toUpperCase()} TRAFFIC` : "READY"}</span></div>
          <div className="duration"><strong>{prediction?.duration_minutes.toFixed(1) ?? "—"}</strong><span>MIN</span></div>
          <div className="result-grid"><div><small>DISTANCE</small><b>{prediction ? `${prediction.distance_km.toFixed(1)} km` : "—"}</b></div>
            <div><small>EST. FARE</small><b>{prediction ? `$${prediction.estimated_fare_usd.toFixed(2)}` : "—"}</b></div>
            <div><small>ENGINE</small><b>{prediction ? (prediction.model_source === "trained model" ? "ML MODEL" : "FALLBACK") : "—"}</b></div></div>
        </div><p className="disclaimer">Planning estimate only · not a quoted fare</p>
      </div>
    </section>
    <section className="method shell" id="method"><div className="section-tag">02 / CRISP-DM</div>
      <div className="method-title"><h2>From raw streets<br />to useful signal.</h2><p>A reproducible six-stage workflow turns messy trip records into a deployment-ready estimate—with monitoring designed in from the start.</p></div>
      <div className="steps">{[
        ["01","Business","Reduce ETA uncertainty for NYC riders and dispatch teams."], ["02","Understand","Profile time, location, vendors, passengers, and target tails."],
        ["03","Prepare","Filter invalid geography and engineer distance, bearing, and cycles."], ["04","Model","Fit regularized regression against a deterministic split."],
        ["05","Evaluate","Compare RMSLE, MAE, R², and slice-level behavior."], ["06","Deploy","Ship one typed Next.js app with health checks and a model card."],
      ].map(([n,title,copy]) => <article key={n}><b>{n}</b><h3>{title}</h3><p>{copy}</p></article>)}</div>
    </section>
    <section className="model-section shell" id="model"><div className="model-copy"><div className="section-tag">03 / MODEL CARD</div><h2>Built to explain<br />its own limits.</h2>
      <p>The target is log-transformed to reduce the influence of rare, very long trips. No drop-off time or post-journey signal enters the feature matrix.</p><a href="/api/model" target="_blank">Inspect model metadata ↗</a></div>
      <div className="metric-panel"><div className="metric-head"><span>ACTIVE MODEL</span><span>{model?.data_source === "synthetic demo" ? "DEMO TRAINED" : model?.status.toUpperCase() ?? "LOADING"}</span></div>
        <h3>{model?.model ?? "—"}</h3><div className="metric-grid"><div><small>RMSLE</small><strong>{model?.metrics.rmsle ?? "N/A"}</strong></div>
          <div><small>MAE</small><strong>{model?.metrics.mae_seconds ? `${model.metrics.mae_seconds}s` : "N/A"}</strong></div><div><small>R²</small><strong>{model?.metrics.r2 ?? "N/A"}</strong></div><div><small>FEATURES</small><strong>{model?.feature_count ?? 16}</strong></div></div>
        <div className="feature-list"><span>Geospatial</span><span>Temporal</span><span>Operational</span><span>Traffic proxy</span></div>
      </div>
    </section>
  </main><footer className="shell"><span>CABWISE / CMPE 255</span><span>Data informs. Humans decide.</span></footer></>;
}
