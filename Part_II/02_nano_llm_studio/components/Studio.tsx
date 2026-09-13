"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import LossChart from "./LossChart";

type Tab = "chat" | "operations" | "data";
type Health = { status: string; device: string; model_loaded: boolean; parameters: number; context_length: number; vocab_size: number; architecture: string[] };
type History = { epoch: number; train_loss: number; validation_loss: number; perplexity: number; learning_rate: number };
type Metrics = { status: string; profile?: string; parameters?: number; model_size_mb?: number; duration_seconds?: number; final?: History; history: History[] };
type Phase = { name: string; status: string; detail: string };
type Message = { role: "user" | "assistant"; text: string; meta?: string };
type Token = { index: number; token_id: number; byte: number; piece: string };

const quickPrompts = ["What is CRISP-DM?", "Explain grouped-query attention.", "What is overfitting?"];

function formatCount(value?: number) {
  if (!value) return "—";
  return value >= 1_000_000 ? `${(value / 1_000_000).toFixed(2)}M` : value.toLocaleString();
}

export default function Studio() {
  const [tab, setTab] = useState<Tab>("chat");
  const [health, setHealth] = useState<Health | null>(null);
  const [metrics, setMetrics] = useState<Metrics>({ status: "untrained", history: [] });
  const [phases, setPhases] = useState<Phase[]>([]);
  const [training, setTraining] = useState<Record<string, unknown>>({ status: "idle" });
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [nextHealth, nextMetrics, crisp, trainingState] = await Promise.all([
        api<Health>("/health"), api<Metrics>("/metrics"), api<{ phases: Phase[] }>("/crisp-dm"), api<Record<string, unknown>>("/training"),
      ]);
      setHealth(nextHealth); setMetrics(nextMetrics); setPhases(crisp.phases); setTraining(trainingState); setError("");
    } catch {
      setError("Python API offline — start it on port 8000.");
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (training.status !== "training" && training.status !== "queued") return;
    const timer = window.setInterval(() => void refresh(), 1500);
    return () => window.clearInterval(timer);
  }, [training.status, refresh]);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">P</div><div><strong>PocketLM</strong><span>MODEL STUDIO</span></div></div>
        <nav aria-label="Main navigation">
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}><span>◌</span> Chat lab</button>
          <button className={tab === "operations" ? "active" : ""} onClick={() => setTab("operations")}><span>⌁</span> Model ops</button>
          <button className={tab === "data" ? "active" : ""} onClick={() => setTab("data")}><span>◇</span> Data lab</button>
        </nav>
        <div className="sidebar-bottom">
          <div className="status-row"><i className={health?.model_loaded ? "online" : "warm"} />{health?.model_loaded ? "Model ready" : "Awaiting training"}</div>
          <small>{health?.device ?? "API offline"} · {health?.context_length ?? 256} context</small>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><p className="eyebrow">LOCAL LANGUAGE INTELLIGENCE</p><h1>{tab === "chat" ? "Conversation lab" : tab === "operations" ? "Model operations" : "Data workbench"}</h1></div>
          <div className="top-actions"><span className="runtime"><i /> {health?.device ?? "disconnected"}</span><button className="icon-button" onClick={() => void refresh()} aria-label="Refresh">↻</button></div>
        </header>
        {error && <div className="notice">{error} Run <code>python -m uvicorn python.api:app --reload</code></div>}
        {tab === "chat" && <ChatLab ready={Boolean(health?.model_loaded)} onOpenOps={() => setTab("operations")} />}
        {tab === "operations" && <Operations health={health} metrics={metrics} phases={phases} training={training} refresh={refresh} />}
        {tab === "data" && <DataLab />}
      </section>
    </main>
  );
}

function ChatLab({ ready, onOpenOps }: { ready: boolean; onOpenOps: () => void }) {
  const [messages, setMessages] = useState<Message[]>([{ role: "assistant", text: "Hello — I’m PocketLM. I’m a tiny model you can train and inspect entirely on this laptop." }]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [temperature, setTemperature] = useState(0.8);

  async function send(event?: FormEvent, preset?: string) {
    event?.preventDefault();
    const message = (preset ?? input).trim();
    if (!message || busy) return;
    setMessages((current) => [...current, { role: "user", text: message }]); setInput(""); setBusy(true);
    try {
      const result = await api<{ reply: string; tokens_generated: number; tokens_per_second: number; latency_ms: number }>("/chat", {
        method: "POST", body: JSON.stringify({ message, temperature, top_p: 0.9, top_k: 40, max_new_tokens: 96 }),
      });
      setMessages((current) => [...current, { role: "assistant", text: result.reply || "[The model emitted no printable bytes.]", meta: `${result.tokens_generated} tokens · ${result.tokens_per_second} tok/s · ${result.latency_ms} ms` }]);
    } catch (caught) {
      setMessages((current) => [...current, { role: "assistant", text: caught instanceof Error ? caught.message : "Generation failed." }]);
    } finally { setBusy(false); }
  }

  return (
    <div className="chat-layout">
      <section className="chat-panel card">
        <div className="chat-head"><div><span className="live-dot" /> Live inference</div><span>Byte tokenizer · KV cache</span></div>
        <div className="messages">
          {messages.map((message, index) => <div className={`message ${message.role}`} key={index}><span className="avatar">{message.role === "assistant" ? "P" : "Y"}</span><div><p>{message.text}</p>{message.meta && <small>{message.meta}</small>}</div></div>)}
          {busy && <div className="message assistant"><span className="avatar">P</span><div className="thinking"><i/><i/><i/></div></div>}
        </div>
        <div className="quick-prompts">{quickPrompts.map((prompt) => <button key={prompt} onClick={() => void send(undefined, prompt)}>{prompt}</button>)}</div>
        <form className="composer" onSubmit={send}>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }} placeholder="Ask the model something…" rows={2} />
          <button disabled={busy || !input.trim()} aria-label="Send message">↑</button>
        </form>
      </section>
      <aside className="chat-settings card">
        <p className="eyebrow">INFERENCE</p><h2>Generation controls</h2>
        <label><span>Temperature <b>{temperature.toFixed(1)}</b></span><input type="range" min="0" max="1.5" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
        <div className="setting-row"><span>Top-p</span><b>0.9</b></div><div className="setting-row"><span>Top-k</span><b>40</b></div><div className="setting-row"><span>Max output</span><b>96</b></div>
        <div className={`readiness ${ready ? "ready" : ""}`}><strong>{ready ? "Checkpoint loaded" : "Training required"}</strong><p>{ready ? "Local weights are ready for inference." : "Train a profile before chatting."}</p>{!ready && <button onClick={onOpenOps}>Open model ops →</button>}</div>
      </aside>
    </div>
  );
}

function Operations({ health, metrics, phases, training, refresh }: { health: Health | null; metrics: Metrics; phases: Phase[]; training: Record<string, unknown>; refresh: () => Promise<void> }) {
  const [profile, setProfile] = useState("nano"); const [epochs, setEpochs] = useState(20); const [starting, setStarting] = useState(false);
  const isTraining = training.status === "training" || training.status === "queued";
  async function start() {
    setStarting(true);
    try { await api("/training", { method: "POST", body: JSON.stringify({ profile, epochs, batch_size: 16, learning_rate: 0.0003 }) }); await refresh(); }
    finally { setStarting(false); }
  }
  return (
    <div className="ops-grid">
      <div className="metric-card"><span>PARAMETERS</span><strong>{formatCount(metrics.parameters ?? health?.parameters)}</strong><small>tied embeddings</small></div>
      <div className="metric-card"><span>VALIDATION PPL</span><strong>{metrics.final?.perplexity ?? "—"}</strong><small>held-out split</small></div>
      <div className="metric-card"><span>MODEL SIZE</span><strong>{metrics.model_size_mb ? `${metrics.model_size_mb} MB` : "—"}</strong><small>PyTorch checkpoint</small></div>
      <div className="metric-card"><span>TRAIN TIME</span><strong>{metrics.duration_seconds ? `${metrics.duration_seconds}s` : "—"}</strong><small>on {health?.device ?? "local device"}</small></div>
      <section className="card telemetry"><div className="section-head"><div><p className="eyebrow">EVALUATION</p><h2>Learning telemetry</h2></div><span className="tag">{metrics.profile ?? "no run"}</span></div><LossChart points={metrics.history ?? []} /></section>
      <section className="card training-card"><p className="eyebrow">MODELING</p><h2>Train a checkpoint</h2><p className="muted">Runs in the Python service and automatically activates the best available local checkpoint.</p>
        <label>Compute profile<select value={profile} onChange={(event) => setProfile(event.target.value)}><option value="micro">Micro · fastest</option><option value="nano">Nano · balanced</option></select></label>
        <label>Epochs<input type="number" min="1" max="100" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /></label>
        {isTraining && <div className="progress"><span style={{ width: `${(Number(training.epoch ?? 0) / Number(training.epochs ?? epochs)) * 100}%` }} /></div>}
        <button className="primary" disabled={starting || isTraining} onClick={() => void start()}>{isTraining ? `Training epoch ${training.epoch ?? 0}/${training.epochs ?? epochs}` : "Start local training"}</button>
      </section>
      <section className="card crisp-card"><div className="section-head"><div><p className="eyebrow">LIFECYCLE</p><h2>CRISP-DM control plane</h2></div><span className="tag">6 phases</span></div><div className="phases">{phases.map((phase, index) => <div className="phase" key={phase.name}><span>{index + 1}</span><div><strong>{phase.name}</strong><p>{phase.detail}</p></div><i className={phase.status}>{phase.status}</i></div>)}</div></section>
      <section className="card primitives"><p className="eyebrow">ARCHITECTURE</p><h2>Modern, intentionally small</h2><div>{health?.architecture.map((item) => <span key={item}>{item}</span>)}</div><p className="muted">Four decoder blocks, shared embeddings, 2 KV heads, and a 256-token context keep memory use laptop-friendly.</p></section>
    </div>
  );
}

function DataLab() {
  const [text, setText] = useState("Small models make ideas tangible."); const [tokens, setTokens] = useState<Token[]>([]);
  useEffect(() => { const timer = window.setTimeout(() => { void api<{ tokens: Token[] }>("/tokenize", { method: "POST", body: JSON.stringify({ text }) }).then((result) => setTokens(result.tokens)).catch(() => setTokens([])); }, 180); return () => clearTimeout(timer); }, [text]);
  return <div className="data-grid"><section className="card tokenizer"><p className="eyebrow">DATA PREPARATION</p><h2>Byte tokenizer inspector</h2><p className="muted">Every UTF-8 byte maps to one stable ID, so there is no unknown token.</p><textarea value={text} onChange={(event) => setText(event.target.value)} rows={4} /><div className="token-summary"><strong>{tokens.length}</strong> tokens <span>{text.length} characters</span></div><div className="token-list">{tokens.map((token) => <div key={token.index}><span>{token.piece.trim() || "·"}</span><b>{token.token_id}</b><small>0x{token.byte.toString(16).padStart(2, "0")}</small></div>)}</div></section><section className="card dataset-card"><p className="eyebrow">DATA UNDERSTANDING</p><h2>Corpus contract</h2><div className="contract"><span>FORMAT</span><strong>JSON Lines</strong><span>FIELDS</span><strong>system · user · assistant</strong><span>OBJECTIVE</span><strong>Assistant-only causal loss</strong><span>SPLIT</span><strong>85% train · 15% validation</strong></div><div className="callout"><strong>Why bytes?</strong><p>A 262-token vocabulary is tiny, multilingual-safe, transparent, and ideal for demonstrating the full LLM pipeline on modest hardware.</p></div></section></div>;
}
