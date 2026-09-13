import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cabwise — NYC journey intelligence",
  description: "NYC taxi trip duration prediction and CRISP-DM model dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
