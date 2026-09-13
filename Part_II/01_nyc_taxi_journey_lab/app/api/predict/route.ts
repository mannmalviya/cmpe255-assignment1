import { NextRequest, NextResponse } from "next/server";
import { predictTrip, TripInput } from "@/lib/model";

function isValid(input: Partial<TripInput>): input is TripInput {
  return typeof input.pickup_datetime === "string" && !Number.isNaN(Date.parse(input.pickup_datetime))
    && typeof input.pickup_longitude === "number" && input.pickup_longitude >= -74.15 && input.pickup_longitude <= -73.70
    && typeof input.dropoff_longitude === "number" && input.dropoff_longitude >= -74.15 && input.dropoff_longitude <= -73.70
    && typeof input.pickup_latitude === "number" && input.pickup_latitude >= 40.55 && input.pickup_latitude <= 40.95
    && typeof input.dropoff_latitude === "number" && input.dropoff_latitude >= 40.55 && input.dropoff_latitude <= 40.95
    && Number.isInteger(input.passenger_count) && input.passenger_count! >= 1 && input.passenger_count! <= 6
    && (input.vendor_id === 1 || input.vendor_id === 2);
}

export async function POST(request: NextRequest) {
  let input: Partial<TripInput>;
  try { input = await request.json(); }
  catch { return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 }); }
  if (!isValid(input)) return NextResponse.json({ detail: "Trip fields are missing or outside NYC bounds" }, { status: 422 });
  return NextResponse.json(predictTrip(input));
}
