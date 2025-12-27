import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LM Feature Studio",
  description: "Gemma-3 SAE Analysis and Steering System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen">{children}</body>
    </html>
  );
}
