import { NextResponse } from "next/server";
import { artifact, featureNames } from "@/lib/model";

export function GET() {
  return NextResponse.json({
    status: "trained", model: artifact.model, metrics: artifact.metrics,
    training_rows: artifact.training_rows, validation_rows: artifact.validation_rows,
    feature_count: featureNames.length, trained_at: artifact.trained_at,
    data_source: artifact.data_source,
  });
}

