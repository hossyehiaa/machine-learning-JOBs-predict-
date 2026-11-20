import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Career Intelligence Studio — AI Job Matching & Career Pathing",
  description:
    "AI-powered resume-to-job matching, skill gap analysis, and career trajectory planning.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased selection:bg-zinc-200 selection:text-zinc-950">
        {children}
      </body>
    </html>
  );
}
