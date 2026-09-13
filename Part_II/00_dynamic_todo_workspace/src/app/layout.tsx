import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tempo — Make space for what matters",
  description: "A calm, focused workspace for planning and completing your day.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: `try{document.documentElement.dataset.theme=localStorage.getItem('tempo-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}catch(e){}` }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
