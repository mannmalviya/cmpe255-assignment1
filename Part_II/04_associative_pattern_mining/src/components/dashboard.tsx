"use client";

import { useMemo, useState } from "react";
import {
  Activity, ArrowRight, BarChart3, BookOpen, Boxes, Check, ChevronRight, CircleDot,
  Database, FlaskConical, Gauge, LayoutDashboard, Menu, Network, PackagePlus, Search,
  ShieldCheck, ShoppingBasket, Sparkles, Timer, TrendingUp, X,
} from "lucide-react";
import { crisp, experiments, products, rules, topItems, type Rule } from "@/lib/data";

type View = "overview" | "rules" | "basket" | "admin" | "crisp";
const nav = [
  ["overview", "Overview", LayoutDashboard], ["rules", "Rule explorer", Network],
  ["basket", "Basket lab", ShoppingBasket], ["admin", "Model admin", FlaskConical],
  ["crisp", "CRISP-DM", BookOpen],
] as const;

export function Dashboard() {
  const [view, setView] = useState<View>("overview");
  const [mobile, setMobile] = useState(false);
  return <div className="shell">
    <aside className={mobile ? "open" : ""}>
      <div className="brand"><div><CircleDot size={21} /></div><p><b>BasketLens</b><span>Pattern intelligence</span></p></div>
      <nav><label>ANALYTICS WORKSPACE</label>{nav.map(([id, label, Icon]) => <button key={id} className={view === id ? "active" : ""} onClick={() => { setView(id); setMobile(false); }}><Icon size={17} />{label}{view === id && <ChevronRight className="push" size={15} />}</button>)}</nav>
      <div className="dataset-card"><span><Database size={17} /></span><p>Instacart Market Basket</p><b>5,000 preview orders</b><small>Deterministic demo artifact</small><i><em /> Contract passed</i></div>
      <footer><span>PIPELINE STATUS</span><p><i /> Ready for Kaggle CSVs</p><small>Artifact v1 · ECLAT champion</small></footer>
    </aside>
    <main>
      <header><button className="menu" onClick={() => setMobile(!mobile)} aria-label="Toggle menu"><Menu size={19} /></button><div className="search"><Search size={16} /><span>Search products, rules, metrics...</span><kbd>⌘ K</kbd></div><div className="source-pill"><i /> DEMO DATA</div><div className="avatar">DS</div></header>
      <div className="content">
        {view === "overview" && <Overview onNavigate={setView} />}
        {view === "rules" && <RuleExplorer />}
        {view === "basket" && <BasketLab />}
        {view === "admin" && <Admin />}
        {view === "crisp" && <Crisp />}
      </div>
    </main>
  </div>;
}

function Title({ crumb, title, copy, action }: { crumb: string; title: string; copy: string; action?: React.ReactNode }) {
  return <div className="title"><div><span>{crumb}</span><h1>{title}</h1><p>{copy}</p></div>{action}</div>;
}
function PanelHead({ title, copy, tag }: { title: string; copy: string; tag?: string }) {
  return <div className="panel-head"><div><h2>{title}</h2><p>{copy}</p></div>{tag && <span>{tag}</span>}</div>;
}

function Overview({ onNavigate }: { onNavigate: (view: View) => void }) {
  return <>
    <Title crumb="MARKET BASKET / OVERVIEW" title="Patterns hiding in plain sight." copy="From thousands of orders to explainable product affinities your merchandising team can use." action={<button className="primary" onClick={() => onNavigate("basket")}><ShoppingBasket size={16} /> Build a basket</button>} />
    <section className="kpis">
      <Kpi icon={ShoppingBasket} label="Orders analyzed" value="5,000" note="Preview sample" />
      <Kpi icon={Boxes} label="Frequent itemsets" value="78" note="At 3.5% support" />
      <Kpi icon={Network} label="Qualified rules" value="203" note="Lift ≥ 1.20" />
      <Kpi icon={TrendingUp} label="Strongest lift" value="4.45×" note="Pasta → parmesan" accent />
    </section>
    <section className="overview-grid">
      <div className="panel network-panel"><PanelHead title="Product affinity network" copy="Node size reflects support · links show strongest rules" tag="16 PRODUCTS" /><AffinityNetwork /></div>
      <div className="panel frequency"><PanelHead title="Basket leaders" copy="Most frequently observed products" />
        <div className="bars-list">{topItems.map(([name, count, support], index) => <div key={name}><span className="rank">0{index + 1}</span><p><b>{name}</b><small>{count.toLocaleString()} orders</small></p><div className="hbar"><i style={{ width: `${support / 36 * 100}%` }} /></div><strong>{support}%</strong></div>)}</div>
      </div>
    </section>
    <section className="panel rule-preview"><PanelHead title="High-opportunity rules" copy="Ranked by lift, then confidence" tag="TOP 3" /><div className="rule-cards">{rules.slice(0, 3).map((rule, i) => <RuleCard rule={rule} key={i} />)}</div></section>
  </>;
}

function Kpi({ icon: Icon, label, value, note, accent }: { icon: typeof Activity; label: string; value: string; note: string; accent?: boolean }) {
  return <div className={`kpi ${accent ? "accent" : ""}`}><div><span>{label}</span><Icon size={17} /></div><b>{value}</b><p><TrendingUp size={13} />{note}</p></div>;
}

const graphNodes = [
  ["Avocado", 43, 43, 18, "#80d49c"], ["Limes", 27, 27, 12, "#b7dc67"], ["Cilantro", 17, 48, 10, "#70bd89"], ["Chips", 31, 67, 12, "#f0bd61"], ["Salsa", 49, 75, 13, "#e9935d"],
  ["Pasta", 72, 25, 12, "#dcae62"], ["Marinara", 86, 38, 12, "#e97d68"], ["Parmesan", 72, 51, 13, "#e9d070"], ["Bread", 59, 71, 17, "#d3a56b"], ["PB", 79, 75, 11, "#bc8659"], ["Jam", 91, 66, 11, "#df7d91"], ["Milk", 48, 18, 17, "#8ec7dd"], ["Eggs", 37, 10, 12, "#ead884"], ["Bananas", 17, 12, 12, "#e4d957"],
] as const;
const edges = [[0,1],[0,2],[0,3],[1,2],[1,3],[3,4],[5,6],[5,7],[6,7],[8,9],[8,10],[9,10],[11,12],[11,13],[12,13]] as const;
function AffinityNetwork() {
  return <svg className="network" viewBox="0 0 100 85" role="img" aria-label="Network of product associations">{edges.map(([a,b], i) => <line key={i} x1={graphNodes[a][1]} y1={graphNodes[a][2]} x2={graphNodes[b][1]} y2={graphNodes[b][2]} />)}{graphNodes.map(([name,x,y,r,color]) => <g key={name}><circle cx={x} cy={y} r={r / 3.2} fill={color} /><circle className="halo" cx={x} cy={y} r={r / 2.45} /><text x={x} y={y + r / 1.7}>{name}</text></g>)}</svg>;
}

function RuleCard({ rule }: { rule: Rule }) {
  return <div className="rule-card"><div className="rule-flow"><span>{rule.left.join(" + ")}</span><ArrowRight size={16} /><b>{rule.right.join(" + ")}</b></div><div className="rule-metrics"><Metric label="SUPPORT" value={`${(rule.support * 100).toFixed(1)}%`} /><Metric label="CONFIDENCE" value={`${(rule.confidence * 100).toFixed(1)}%`} /><Metric label="LIFT" value={`${rule.lift.toFixed(2)}×`} strong /></div></div>;
}
function Metric({ label, value, strong }: { label: string; value: string; strong?: boolean }) { return <div className={strong ? "strong" : ""}><span>{label}</span><b>{value}</b></div>; }

function RuleExplorer() {
  const [query, setQuery] = useState("");
  const [lift, setLift] = useState(1.2);
  const filtered = rules.filter(rule => rule.lift >= lift && [...rule.left, ...rule.right].join(" ").toLowerCase().includes(query.toLowerCase()));
  return <>
    <Title crumb="DISCOVERY / RULE EXPLORER" title="Inspect every recommendation." copy="Filter associations and audit the evidence behind each potential cross-sell." />
    <div className="filterbar"><label><Search size={16} /><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Find a product..." /></label><div><span>Minimum lift</span><input type="range" min="1.2" max="4.5" step=".1" value={lift} onChange={e => setLift(Number(e.target.value))} /><b>{lift.toFixed(1)}×</b></div><span className="result-count">{filtered.length} RULES</span></div>
    <section className="panel table-panel"><table><thead><tr><th>Antecedent</th><th></th><th>Consequent</th><th>Support</th><th>Confidence</th><th>Lift</th><th>Conviction</th></tr></thead><tbody>{filtered.map((r, i) => <tr key={i}><td>{r.left.join(" + ")}</td><td><ArrowRight size={14} /></td><td><b>{r.right.join(" + ")}</b></td><td>{(r.support * 100).toFixed(1)}%</td><td>{(r.confidence * 100).toFixed(1)}%</td><td><strong>{r.lift.toFixed(2)}×</strong></td><td>{r.conviction.toFixed(2)}</td></tr>)}</tbody></table>{!filtered.length && <div className="empty">No rules meet these filters.</div>}</section>
  </>;
}

function BasketLab() {
  const [basket, setBasket] = useState<string[]>(["Marinara Sauce", "Penne Pasta"]);
  const [searched, setSearched] = useState("");
  const recommendations = useMemo(() => rules.filter(rule => rule.left.every(item => basket.includes(item))).flatMap(rule => rule.right.filter(item => !basket.includes(item)).map(item => ({ item, rule }))).filter((value, index, all) => all.findIndex(x => x.item === value.item) === index).slice(0, 4), [basket]);
  const add = (item: string) => setBasket(current => current.includes(item) ? current : [...current, item]);
  return <>
    <Title crumb="DECISION TOOLS / BASKET LAB" title="Build a basket. Reveal the next best item." copy="Simulate the production recommender using transparent association rules." />
    <section className="basket-layout">
      <div className="panel catalog"><PanelHead title="Product catalog" copy="Choose one or more basket items" tag={`${products.length} ITEMS`} /><label><Search size={15} /><input value={searched} onChange={e => setSearched(e.target.value)} placeholder="Search catalog" /></label><div className="product-list">{products.filter(p => p.toLowerCase().includes(searched.toLowerCase())).map(product => <button key={product} disabled={basket.includes(product)} onClick={() => add(product)}><span><PackagePlus size={15} /></span><p>{product}<small>{basket.includes(product) ? "In basket" : "Add product"}</small></p><b>+</b></button>)}</div></div>
      <div className="basket-workspace"><div className="basket-head"><span><ShoppingBasket size={18} /></span><div><b>ACTIVE BASKET</b><small>{basket.length} items selected</small></div><button onClick={() => setBasket([])}>Clear</button></div><div className="basket-items">{basket.length ? basket.map(item => <button key={item} onClick={() => setBasket(basket.filter(x => x !== item))}>{item}<X size={13} /></button>) : <p>Add a product to begin.</p>}</div>
        <div className="recommendations"><span className="section-label"><Sparkles size={14} /> RECOMMENDED ADD-ONS</span>{recommendations.length ? recommendations.map(({ item, rule }, i) => <div className="recommendation" key={item}><span className="rec-rank">0{i + 1}</span><div><b>{item}</b><small>Triggered by {rule.left.join(" + ")}</small></div><Metric label="CONFIDENCE" value={`${(rule.confidence * 100).toFixed(1)}%`} /><Metric label="LIFT" value={`${rule.lift.toFixed(2)}×`} strong /><button onClick={() => add(item)}>Add</button></div>) : <div className="no-rec"><Network size={25} /><b>No qualifying rule yet</b><span>Try Pasta + Marinara, Eggs + Bananas, or Bread + Peanut Butter.</span></div>}</div>
        <div className="latency"><Timer size={15} /><span>Precomputed lookup</span><b>&lt; 1 ms</b><i>READY</i></div>
      </div>
    </section>
  </>;
}

function Admin() {
  const max = Math.max(...experiments.map(x => x.rules));
  return <>
    <Title crumb="DATA SCIENCE ADMIN / MODEL HEALTH" title="Every mining decision, visible." copy="Compare algorithms, tune pruning thresholds, and review deployment gates." action={<button className="secondary"><Gauge size={16} /> Artifact v1.0</button>} />
    <div className="success"><Check size={18} /><div><b>Candidate artifact passed validation</b><span>Apriori and ECLAT produced identical frequent-itemset support values.</span></div><strong>READY</strong></div>
    <section className="admin-grid"><div className="panel leaderboard"><PanelHead title="Algorithm benchmark" copy="Same baskets · same 3.5% support" tag="2 CANDIDATES" /><div className="model-row head"><span>Rank</span><span>Algorithm</span><span>Runtime</span><span>Itemsets</span><span>Role</span></div><div className="model-row"><span>01</span><b>ECLAT</b><strong>5.61 ms</strong><span>78</span><i>CHAMPION</i></div><div className="model-row"><span>02</span><b>Apriori</b><span>104.34 ms</span><span>78</span><em>BASELINE</em></div></div>
      <div className="panel experiment"><PanelHead title="Support sensitivity" copy="Rule count by threshold" tag="SELECTED: 3.5%" /><div className="vertical-bars">{experiments.map(row => <div key={row.support} className={row.support === .035 ? "chosen" : ""}><span>{row.rules}</span><i style={{ height: `${row.rules / max * 100}%` }} /><b>{row.support * 100}%</b></div>)}</div></div></section>
    <section className="panel governance"><PanelHead title="Deployment governance" copy="Checks required before promoting a new rule artifact" /><div>{[[Database,"Schema contract","Order and product keys valid"],[ShieldCheck,"Rule parity","Support values reconciled"],[Activity,"Business review","Top 25 rules inspected"],[Gauge,"Drift baseline","Coverage snapshot versioned"]].map(([Icon,title,copy]) => { const C = Icon as typeof Database; return <article key={String(title)}><span><C size={16} /></span><p><b>{String(title)}</b><small>{String(copy)}</small></p><Check size={14} /></article>; })}</div></section>
  </>;
}

function Crisp() {
  const [selected, setSelected] = useState(0);
  const phase = crisp[selected];
  return <>
    <Title crumb="METHODOLOGY / CRISP-DM" title="From basket data to deployed decisions." copy="A six-phase, auditable workflow for responsible association pattern mining." />
    <section className="crisp-layout"><div className="crisp-nav">{crisp.map((item, i) => <button key={item.n} className={i === selected ? "active" : ""} onClick={() => setSelected(i)}><span>{item.n}</span><p><b>{item.name}</b><small>{i < 5 ? "Complete" : "Ready"}</small></p><ChevronRight size={15} /></button>)}</div><div className="crisp-content"><span>PHASE {phase.n} / 06</span><h2>{phase.name}</h2><p className="lead">{phase.note}</p><div className="decision"><Sparkles size={20} /><p><b>Decision record</b><span>{phase.decision}</span></p></div><h3>Phase deliverables</h3><ul>{phase.items.map(item => <li key={item}><Check size={14} />{item}</li>)}</ul><div className="phase-foot"><span>{selected + 1} of 6 phases</span>{selected < 5 && <button className="primary" onClick={() => setSelected(selected + 1)}>Next phase <ChevronRight size={15} /></button>}</div></div></section>
  </>;
}
