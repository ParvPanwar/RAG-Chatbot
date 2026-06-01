import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Video Compare Desk",
  description: "Compare two YouTube videos or two Instagram Reels with RAG chat.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
