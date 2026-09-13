import { describe, expect, it } from "vitest";
import { addDaysKey, friendlyDate, localDateKey } from "@/lib/date";

describe("local date helpers", () => {
  const noon = new Date(2026, 8, 13, 12);

  it("formats dates without UTC drift", () => {
    expect(localDateKey(noon)).toBe("2026-09-13");
    expect(addDaysKey(1, noon)).toBe("2026-09-14");
  });

  it("uses human labels for nearby dates", () => {
    expect(friendlyDate("2026-09-13", noon)).toBe("Today");
    expect(friendlyDate("2026-09-14", noon)).toBe("Tomorrow");
    expect(friendlyDate(null, noon)).toBeNull();
  });
});

