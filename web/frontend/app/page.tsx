"use client";

import { useState, useEffect, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import ChatArea from "@/components/ChatArea";
import ThemeToggle from "@/components/ThemeToggle";
import { fetchDocuments, fetchStats, Document, GraphStats } from "@/lib/api";

export default function Home() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      const [docs, graphStats] = await Promise.all([fetchDocuments(), fetchStats()]);
      setDocuments(docs);
      setStats(graphStats);
      setError(null);
    } catch {
      setError("Cannot reach API. Make sure the FastAPI server is running on port 8000.");
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30_000);
    return () => clearInterval(interval);
  }, [loadData]);

  return (
    <main
      style={{
        display: "flex",
        height: "100vh",
        width: "100vw",
        overflow: "hidden",
        background: "var(--bg-primary)",
        position: "relative",
      }}
    >
      {/* Error banner */}
      {error && (
        <div
          style={{
            position: "fixed",
            top: "12px",
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 100,
            background: "var(--bg-card)",
            border: "1px solid rgba(220,60,60,0.35)",
            borderRadius: "8px",
            padding: "9px 18px",
            fontSize: "12px",
            color: "#c0392b",
            boxShadow: "var(--shadow-lg)",
            maxWidth: "480px",
            textAlign: "center",
            fontWeight: 500,
          }}
        >
          ! {error}
        </div>
      )}

      <div style={{ display: "flex", width: "100%", height: "100%", position: "relative" }}>
        <Sidebar
          documents={documents}
          onDocumentsChange={loadData}
          stats={stats}
          themeToggle={<ThemeToggle />}
        />
        <ChatArea />
      </div>
    </main>
  );
}
