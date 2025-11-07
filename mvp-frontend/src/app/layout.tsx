import "./globals.css";
import { Inter } from "next/font/google";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "NFL Model",
  description: "Picks, probabilities & history",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-50 text-gray-900`}>
        <header className="border-b bg-white">
          <nav className="max-w-6xl mx-auto px-4 sm:px-6">
            <div className="h-14 flex items-center justify-between">
              <Link href="/" className="font-semibold tracking-tight">
                NFL Model
              </Link>
              <div className="flex items-center gap-4 text-sm">
                <Link className="hover:text-black/70" href="/">Home</Link>
                <Link className="hover:text-black/70" href="/history">History</Link>
                <a
                  className="hidden sm:inline-block text-gray-500 hover:text-gray-700"
                  href={process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000/health"}
                  target="_blank"
                >
                  API Health
                </a>
              </div>
            </div>
          </nav>
        </header>
        <main className="max-w-6xl mx-auto p-4 sm:p-6">{children}</main>
        <footer className="py-8 text-center text-xs text-gray-500">
          Built with Next.js + FastAPI
        </footer>
      </body>
    </html>
  );
}
