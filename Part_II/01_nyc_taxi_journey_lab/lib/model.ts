import artifactData from "@/models/taxi_duration.json";

export const featureNames = [
  "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
  "passenger_count", "vendor_id", "distance_km", "manhattan_km", "bearing_sin",
  "bearing_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
  "is_rush_hour",
] as const;

export type TripInput = {
  pickup_longitude: number; pickup_latitude: number;
  dropoff_longitude: number; dropoff_latitude: number;
  pickup_datetime: string; passenger_count: number; vendor_id: number;
};

type Artifact = {
  model: string; trained_at: string; training_rows: number; validation_rows: number;
  data_source: string; metrics: Record<string, number>; target_transform: string;
  intercept: number; coefficients: number[]; feature_means: number[]; feature_scales: number[];
};

export const artifact = artifactData as Artifact;
const earthKm = 6371.0088;
const radians = (value: number) => value * Math.PI / 180;

function distance(lat1: number, lon1: number, lat2: number, lon2: number) {
  const lat1r = radians(lat1), lat2r = radians(lat2);
  const dlat = lat2r - lat1r, dlon = radians(lon2 - lon1);
  const a = Math.sin(dlat / 2) ** 2 + Math.cos(lat1r) * Math.cos(lat2r) * Math.sin(dlon / 2) ** 2;
  return 2 * earthKm * Math.asin(Math.sqrt(a));
}

export function buildFeatures(input: TripInput) {
  const { pickup_latitude: lat1, pickup_longitude: lon1, dropoff_latitude: lat2, dropoff_longitude: lon2 } = input;
  const distanceKm = distance(lat1, lon1, lat2, lon2);
  const manhattanKm = distance(lat1, lon1, lat2, lon1) + distance(lat1, lon1, lat1, lon2);
  const dlon = radians(lon2 - lon1);
  const y = Math.sin(dlon) * Math.cos(radians(lat2));
  const x = Math.cos(radians(lat1)) * Math.sin(radians(lat2)) - Math.sin(radians(lat1)) * Math.cos(radians(lat2)) * Math.cos(dlon);
  const bearing = Math.atan2(y, x);
  const [datePart, timePart = "00:00"] = input.pickup_datetime.split("T");
  const [hourValue, minuteValue] = timePart.split(":").map(Number);
  const hour = hourValue + minuteValue / 60;
  const jsDay = new Date(`${datePart}T00:00:00Z`).getUTCDay();
  const dow = (jsDay + 6) % 7;
  const rush = (hour >= 7 && hour <= 10) || (hour >= 16 && hour <= 19);
  const values = [lon1, lat1, lon2, lat2, input.passenger_count, input.vendor_id,
    distanceKm, manhattanKm, Math.sin(bearing), Math.cos(bearing),
    Math.sin(2 * Math.PI * hour / 24), Math.cos(2 * Math.PI * hour / 24),
    Math.sin(2 * Math.PI * dow / 7), Math.cos(2 * Math.PI * dow / 7),
    dow >= 5 ? 1 : 0, rush ? 1 : 0];
  return { values, distanceKm, rush, hour };
}

export function predictTrip(input: TripInput) {
  const { values, distanceKm, rush, hour } = buildFeatures(input);
  const logSeconds = values.reduce((sum, value, index) => sum +
    ((value - artifact.feature_means[index]) / artifact.feature_scales[index]) * artifact.coefficients[index], artifact.intercept);
  const seconds = Math.max(60, Math.min(7200, Math.round(Math.expm1(logSeconds))));
  const minutes = seconds / 60;
  return {
    duration_seconds: seconds, duration_minutes: Number(minutes.toFixed(1)),
    distance_km: Number(distanceKm.toFixed(2)),
    estimated_fare_usd: Number((3 + distanceKm * 1.85 + minutes * .48 + (rush ? 2.5 : 0)).toFixed(2)),
    traffic_band: rush ? "Heavy" : (hour >= 7 && hour <= 21 ? "Moderate" : "Light"),
    model_source: "trained model",
  };
}

