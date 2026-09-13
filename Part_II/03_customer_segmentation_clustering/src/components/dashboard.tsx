"use client";

import { useMemo, useState } from "react";
import {
  Activity, ArrowDownRight, ArrowUpRight, BarChart3, BookOpen, BrainCircuit,
  Check, ChevronRight, CircleDot, Database, FlaskConical, LayoutDashboard,
  Menu, RefreshCw, Search, Sparkles, Target, Users, X,
} from "lucide-react";
import { crispPhases, kScores, modelMetrics, personas, points, type Persona } from "@/lib/data";

type View = "overview" | "segments" | "lab" | "crisp";

const nav: { id: View; label: string; icon: typeof Activity }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "segments", label: "Segments", icon: Users },
  { id: "lab", label: "Model lab", icon: FlaskConical },
  { id: "crisp", label: "CRISP-DM", icon: BookOpen },
];

export function Dashboard() {
  const [view, setView] = useState<View>("overview");
  const [selected, setSelected] = useState(0);
  const [mobileNav, setMobileNav] = useState(false);
  const [predictionOpen, setPredictionOpen] = useState(false);

  return (
    <div className="shell">
      <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
        <div className="brand"><div className="brand-mark"><CircleDot size={21} /></div><div><b>SegmentIQ</b><span>Decision workspace</span></div></div>
        <nav>
          <p className="nav-label">Workspace</p>
          {nav.map(({ id, label, icon: Icon }) => (
            <button key={id} className={view === id ? "active" : ""} onClick={() => { setView(id); setMobileNav(false); }}>
              <Icon size={18} />{label}{view === id && <ChevronRight size={15} className="nav-arrow" />}
            </button>
          ))}
        </nav>
        <div className="data-card">
          <div className="data-icon"><Database size={17} /></div>
          <p>Customer Personality</p><b>2,240 rows</b><span>Kaggle · marketing_campaign.csv</span>
          <div className="quality"><i /><small>Data quality 96.8%</small></div>
        </div>
        <div className="side-footer"><span>MODEL STATUS</span><p><i /> Production ready</p><small>Trained on demo artifact</small></div>
      </aside>

      <main>
        <header>
          <button className="menu-btn" onClick={() => setMobileNav(!mobileNav)} aria-label="Toggle navigation"><Menu size={20} /></button>
          <div className="search"><Search size={17} /><span>Search customers or metrics...</span><kbd>⌘ K</kbd></div>
          <div className="header-actions"><span className="demo-pill">DEMO DATA</span><button className="icon-btn" aria-label="Refresh"><RefreshCw size={17} /></button><div className="avatar">DS</div></div>
        </header>

        <div className="content">
          {view === "overview" && <Overview selected={selected} setSelected={setSelected} onPredict={() => setPredictionOpen(true)} />}
          {view === "segments" && <Segments selected={selected} setSelected={setSelected} />}
          {view === "lab" && <ModelLab />}
          {view === "crisp" && <CrispReport />}
        </div>
      </main>
      {predictionOpen && <PredictionModal onClose={() => setPredictionOpen(false)} />}
    </div>
  );
}

function PageTitle({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return <div className="page-title"><div><span>{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

function Overview({ selected, setSelected, onPredict }: { selected: number; setSelected: (v: number) => void; onPredict: () => void }) {
  return <>
    <PageTitle eyebrow="CUSTOMER INTELLIGENCE / OVERVIEW" title="A clearer view of every customer." description="Behavioral segments turned into decisions your marketing team can act on." action={<button className="primary" onClick={onPredict}><Sparkles size={17} /> Classify a customer</button>} />
    <section className="kpi-grid">
      <Kpi icon={Users} label="Customers analyzed" value="2,240" trend="100% profiled" positive />
      <Kpi icon={Target} label="Discovered segments" value="5" trend="Optimal at k = 5" positive />
      <Kpi icon={Activity} label="Silhouette score" value="0.341" trend="+12.9% vs baseline" positive />
      <Kpi icon={BrainCircuit} label="PCA variance" value="71.4%" trend="Across first 2 components" />
    </section>
    <section className="overview-grid">
      <div className="panel scatter-panel">
        <PanelHead title="Customer landscape" subtitle="PCA projection · click a segment to isolate" tag="PC1 + PC2" />
        <Scatter selected={selected} setSelected={setSelected} />
      </div>
      <div className="panel distribution-panel">
        <PanelHead title="Segment distribution" subtitle="Share of customer base" />
        <div className="donut-wrap"><Donut /><div className="donut-center"><b>2,240</b><span>customers</span></div></div>
        <div className="legend">{personas.map(p => <button key={p.id} onClick={() => setSelected(p.id)} className={selected === p.id ? "selected" : ""}><i style={{ background: p.color }} /><span>{p.name}</span><b>{p.share}%</b></button>)}</div>
      </div>
    </section>
    <section className="panel persona-strip">
      <PanelHead title="Segment playbook" subtitle="Prioritized audiences and recommended actions" tag="5 ACTIVE PERSONAS" />
      <div className="persona-grid">{personas.map(p => <PersonaCard key={p.id} persona={p} active={selected === p.id} onClick={() => setSelected(p.id)} />)}</div>
    </section>
  </>;
}

function Kpi({ icon: Icon, label, value, trend, positive }: { icon: typeof Activity; label: string; value: string; trend: string; positive?: boolean }) {
  return <div className="kpi"><div className="kpi-top"><span>{label}</span><div><Icon size={17} /></div></div><b>{value}</b><p className={positive ? "up" : ""}>{positive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}{trend}</p></div>;
}

function PanelHead({ title, subtitle, tag }: { title: string; subtitle: string; tag?: string }) {
  return <div className="panel-head"><div><h2>{title}</h2><p>{subtitle}</p></div>{tag && <span>{tag}</span>}</div>;
}

function Scatter({ selected, setSelected }: { selected: number; setSelected: (v: number) => void }) {
  return <div className="scatter" role="img" aria-label="PCA scatter plot of customer segments">
    <div className="axis y">PC2</div><div className="axis x">PC1</div>
    {[25, 50, 75].map(n => <span key={`v${n}`} className="grid-v" style={{ left: `${n}%` }} />)}
    {[25, 50, 75].map(n => <span key={`h${n}`} className="grid-h" style={{ top: `${n}%` }} />)}
    {points.map((p, i) => <button key={i} aria-label={`Customer ${p.customer}`} onClick={() => setSelected(p.cluster)} className={`dot ${selected !== p.cluster ? "muted" : ""}`} style={{ left: `${p.x}%`, bottom: `${p.y}%`, background: personas[p.cluster].color }} />)}
    {personas.map(p => { const own = points.filter(x => x.cluster === p.id); const x = own.reduce((a, b) => a + b.x, 0) / own.length; const y = own.reduce((a, b) => a + b.y, 0) / own.length; return <button key={p.id} className={`cluster-label ${selected === p.id ? "active" : ""}`} style={{ left: `${x}%`, bottom: `${y}%`, borderColor: p.color }} onClick={() => setSelected(p.id)}>{p.name.split(" ")[0]}</button>; })}
  </div>;
}

function Donut() {
  const offsets = personas.map((_, index) => personas.slice(0, index).reduce((sum, item) => sum + item.share, 0));
  return <svg className="donut" viewBox="0 0 42 42">{personas.map((p, index) => <circle key={p.id} cx="21" cy="21" r="15.9" fill="transparent" stroke={p.color} strokeWidth="4.4" strokeDasharray={`${p.share} ${100 - p.share}`} strokeDashoffset={25 - offsets[index]} />)}</svg>;
}

function PersonaCard({ persona, active, onClick }: { persona: Persona; active: boolean; onClick: () => void }) {
  return <button className={`persona-card ${active ? "active" : ""}`} style={{ "--accent": persona.color } as React.CSSProperties} onClick={onClick}><div className="persona-number">0{persona.id + 1}</div><i /><h3>{persona.name}</h3><p>{persona.kicker}</p><div><span>{persona.count.toLocaleString()} customers</span><b>{persona.share}%</b></div></button>;
}

function Segments({ selected, setSelected }: { selected: number; setSelected: (v: number) => void }) {
  const p = personas[selected];
  const maxes = { income: 90, spend: 1800, recency: 100, webConversion: 1 };
  return <>
    <PageTitle eyebrow="CUSTOMER INTELLIGENCE / SEGMENTS" title="Know the people behind the clusters." description="Compare customer behaviors and translate patterns into campaign strategy." />
    <div className="segment-tabs">{personas.map(x => <button key={x.id} className={selected === x.id ? "active" : ""} onClick={() => setSelected(x.id)}><i style={{ background: x.color }} />{x.name}</button>)}</div>
    <section className="segment-detail" style={{ "--accent": p.color } as React.CSSProperties}>
      <div className="segment-hero"><span>SEGMENT 0{p.id + 1}</span><h2>{p.name}</h2><p>{p.kicker}</p><div className="segment-count"><b>{p.count}</b><span>customers<br />{p.share}% of base</span></div></div>
      <div className="profile-panel"><h3>Behavior profile</h3>{([
        ["Annual income", p.income, maxes.income, `$${p.income}k`], ["Annual spend", p.spend, maxes.spend, `$${p.spend}`],
        ["Recent activity", 100 - p.recency, 100, `${p.recency} days`], ["Web conversion", p.webConversion, maxes.webConversion, `${Math.round(p.webConversion * 100)}%`],
      ] as [string, number, number, string][]).map(([label, val, max, display]) => <div className="profile-row" key={label}><div><span>{label}</span><b>{display}</b></div><div className="bar"><i style={{ width: `${Math.max(8, val / max * 100)}%` }} /></div></div>)}</div>
      <div className="strategy-panel"><span><Target size={18} /> RECOMMENDED PLAY</span><h3>Turn this segment into momentum.</h3><p>{p.strategy}</p><div className="strategy-tags"><b>Personalized creative</b><b>Segment holdout</b><b>30-day window</b></div></div>
    </section>
    <section className="panel compare-table"><PanelHead title="Cross-segment comparison" subtitle="Core features used for persona interpretation" />
      <div className="table-scroll"><table><thead><tr><th>Segment</th><th>Customers</th><th>Avg. income</th><th>Avg. spend</th><th>Recency</th><th>Web conversion</th></tr></thead><tbody>{personas.map(x => <tr key={x.id}><td><i style={{ background: x.color }} />{x.name}</td><td>{x.count}</td><td>${x.income}k</td><td>${x.spend}</td><td>{x.recency} days</td><td>{Math.round(x.webConversion * 100)}%</td></tr>)}</tbody></table></div>
    </section>
  </>;
}

function ModelLab() {
  const max = Math.max(...kScores.map(x => x.score));
  return <>
    <PageTitle eyebrow="DATA SCIENCE ADMIN / MODEL LAB" title="Every model decision, visible." description="Audit candidate algorithms, cluster selection, and production readiness." action={<button className="secondary"><RefreshCw size={16} /> Retrain pipeline</button>} />
    <div className="admin-alert"><Check size={18} /><div><b>Champion passed deployment gate</b><span>K-Means++ achieved the best balance of separation, stability, and business interpretability.</span></div><span>READY</span></div>
    <section className="lab-grid">
      <div className="panel leaderboard"><PanelHead title="Algorithm leaderboard" subtitle="Ranked by silhouette score" tag="4 CANDIDATES" />
        {modelMetrics.map((m, i) => <div className="model-row" key={m.model}><span className="rank">0{i + 1}</span><div className="model-name"><b>{m.model}</b><span>{m.status}</span></div><Metric label="SILHOUETTE" value={m.silhouette.toFixed(3)} /><Metric label="DB INDEX ↓" value={m.db.toFixed(2)} /><Metric label="CH INDEX" value={m.ch.toFixed(1)} />{i === 0 && <span className="champion">CHAMPION</span>}</div>)}
      </div>
      <div className="panel k-select"><PanelHead title="Cluster selection" subtitle="Silhouette by candidate k" tag="BEST: K = 5" />
        <div className="bars">{kScores.map(x => <div key={x.k} className={x.score === max ? "best" : ""}><span>{x.score.toFixed(3)}</span><i style={{ height: `${x.score / max * 82}%` }} /><b>k{x.k}</b></div>)}</div>
      </div>
    </section>
    <section className="panel governance"><PanelHead title="Production governance" subtitle="Checks required before a new artifact is promoted" />
      <div className="govern-grid"><Govern title="Data contract" note="29 columns validated" /><Govern title="Missing values" note="Income imputed by median" /><Govern title="Stability" note="ARI 0.87 across seeds" /><Govern title="Drift baseline" note="Centroids versioned" /></div>
    </section>
  </>;
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="metric"><span>{label}</span><b>{value}</b></div>; }
function Govern({ title, note }: { title: string; note: string }) { return <div className="govern"><div><Check size={15} /></div><p><b>{title}</b><span>{note}</span></p></div>; }

function CrispReport() {
  const [phase, setPhase] = useState(0);
  return <>
    <PageTitle eyebrow="METHODOLOGY / CRISP-DM" title="From business question to monitored model." description="A complete, auditable six-phase workflow for customer segmentation." />
    <section className="crisp-layout">
      <div className="crisp-nav">{crispPhases.map((p, i) => <button key={p.n} onClick={() => setPhase(i)} className={phase === i ? "active" : ""}><span>{p.n}</span><div><b>{p.name}</b><small>{i < 5 ? "Complete" : "Ready"}</small></div><ChevronRight size={16} /></button>)}</div>
      <div className="crisp-body"><span>PHASE {crispPhases[phase].n} / 06</span><h2>{crispPhases[phase].name}</h2><p className="lead">{crispPhases[phase].note}</p><div className="crisp-callout"><BrainCircuit size={21} /><div><b>Decision record</b><p>{decisionRecords[phase]}</p></div></div><h3>Phase deliverables</h3><ul>{deliverables[phase].map(item => <li key={item}><Check size={15} />{item}</li>)}</ul><div className="phase-footer"><span>{phase + 1} of 6 phases</span>{phase < 5 && <button className="primary" onClick={() => setPhase(phase + 1)}>Next phase <ChevronRight size={16} /></button>}</div></div>
    </section>
  </>;
}

const decisionRecords = [
  "Segmentation is successful only if every cluster maps to a distinct, measurable marketing treatment—not merely a visually separate group.",
  "Income is the only materially incomplete modeling feature. Median imputation is fitted inside the preparation pipeline to prevent leakage.",
  "Raw product spend fields are retained for profiling but consolidated for modeling to reduce redundant distance weighting.",
  "K-Means++ is the champion because it leads internal metrics and yields stable, explainable centroids; GMM remains the probabilistic challenger.",
  "Five clusters are retained after quantitative scoring plus stakeholder review. A higher k adds fragmentation without a distinct action plan.",
  "Inference is artifact-based. Monitor feature missingness, population-share shift, centroid distance, and quarterly business lift.",
];
const deliverables = [
  ["Problem statement and campaign use cases", "Business KPIs and model success criteria", "Risk and constraint register"],
  ["Schema and descriptive statistics", "Missingness and outlier assessment", "Feature distribution review"],
  ["Reproducible cleaning transformer", "Eight behavioral features", "Standardized modeling matrix"],
  ["Four-algorithm benchmark", "Candidate k search from 2 through 8", "PCA projection for interpretation"],
  ["Internal validation scorecard", "Seed stability check", "Persona review and campaign mapping"],
  ["Versioned model bundle", "FastAPI prediction endpoint", "Monitoring and retraining policy"],
];

function PredictionModal({ onClose }: { onClose: () => void }) {
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [income, setIncome] = useState(72);
  const [spend, setSpend] = useState(1100);
  const fallback = useMemo(() => income > 65 && spend > 1000 ? personas[0] : spend > 750 ? personas[1] : income < 40 ? personas[3] : personas[2], [income, spend]);
  const [prediction, setPrediction] = useState({ persona: fallback, confidence: 84, live: false });
  async function classify() {
    setLoading(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8003"}/api/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ age: 40, income: income * 1000, customer_tenure_days: 900, children: 1, total_spend: spend, total_purchases: 12, average_basket: spend / 12, campaign_acceptance: 1, web_conversion: .5 }) });
      if (!response.ok) throw new Error("API unavailable");
      const result = await response.json();
      setPrediction({ persona: personas.find(p => p.name === result.persona_name) ?? fallback, confidence: result.assignment_confidence, live: true });
    } catch {
      setPrediction({ persona: fallback, confidence: 84, live: false });
    } finally { setLoading(false); setSubmitted(true); }
  }
  return <div className="modal-backdrop" onMouseDown={onClose}><div className="modal" onMouseDown={e => e.stopPropagation()}><button className="modal-close" onClick={onClose}><X size={19} /></button><span className="modal-kicker">REAL-TIME PERSONA INFERENCE</span><h2>Classify a customer</h2><p>Try a profile against the production segmentation artifact.</p>{!submitted ? <form onSubmit={e => { e.preventDefault(); void classify(); }}><label>Annual income <span>${income}k</span><input type="range" min="15" max="150" value={income} onChange={e => setIncome(Number(e.target.value))} /></label><label>Annual spend <span>${spend}</span><input type="range" min="50" max="2500" step="50" value={spend} onChange={e => setSpend(Number(e.target.value))} /></label><div className="form-grid"><label>Recency (days)<input defaultValue="28" type="number" /></label><label>Web purchases<input defaultValue="7" type="number" /></label></div><button className="primary" disabled={loading} type="submit"><Sparkles size={16} /> {loading ? "Classifying..." : "Run classification"}</button></form> : <div className="prediction" style={{ "--accent": prediction.persona.color } as React.CSSProperties}><span>MOST LIKELY PERSONA</span><h3>{prediction.persona.name}</h3><p>{prediction.persona.kicker}</p><div><b>{prediction.confidence}%</b><span>assignment confidence</span></div><small>{prediction.live ? "Prediction returned by the trained FastAPI model." : "Demo preview shown because the FastAPI artifact is not running."}</small><button className="secondary" onClick={() => setSubmitted(false)}>Try another profile</button></div>}</div></div>;
}
