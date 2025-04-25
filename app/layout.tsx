import type { Metadata } from "next";
import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";

export const metadata: Metadata = {
  title: "LLM Bias Probe",
  description: "a tool for LLM bias-probing framework",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} dark`}>
      <head>
        <meta 
          name="viewport" 
          content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover" 
        />
      </head>
      <body className={`${GeistSans.className} dark:bg-background touch-none text-base`}>
        {children}
        <Toaster 
          theme="dark" 
          position="top-center"
          className="sm:!bottom-4 sm:!top-auto"
          toastOptions={{
            className: 'sm:!mb-0 sm:!mt-0',
          }}
        />
      </body>
    </html>
  );
}