type Point = { epoch: number; train_loss: number; validation_loss: number };

export default function LossChart({ points }: { points: Point[] }) {
  if (!points.length) {
    return <div className="chart-empty">Loss curves appear after the first training run.</div>;
  }
  const width = 680;
  const height = 220;
  const padding = 26;
  const values = points.flatMap((point) => [point.train_loss, point.validation_loss]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const x = (index: number) => padding + (index / Math.max(1, points.length - 1)) * (width - padding * 2);
  const y = (value: number) => height - padding - ((value - low) / Math.max(0.001, high - low)) * (height - padding * 2);
  const line = (key: "train_loss" | "validation_loss") =>
    points.map((point, index) => `${x(index)},${y(point[key])}`).join(" ");

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training and validation loss by epoch">
        {[0, 1, 2, 3].map((row) => (
          <line key={row} x1={padding} x2={width - padding} y1={padding + row * 55} y2={padding + row * 55} className="grid-line" />
        ))}
        <polyline points={line("train_loss")} className="train-line" />
        <polyline points={line("validation_loss")} className="validation-line" />
      </svg>
      <div className="chart-legend"><span className="train-key" /> Train loss <span className="validation-key" /> Validation loss</div>
    </div>
  );
}

