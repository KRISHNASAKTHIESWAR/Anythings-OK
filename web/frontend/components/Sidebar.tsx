"use client";

import { useState, useRef, useCallback } from "react";
import {
  Document,
  deleteDocument,
  uploadFile,
  getJobStatus,
  JobStatus,
} from "@/lib/api";

interface SidebarProps {
  documents: Document[];
  onDocumentsChange: () => void;
  stats: { entities: number; chunks: number; relationships: number; communities: number } | null;
  themeToggle: React.ReactNode;
}

interface UploadJob {
  job_id: string;
  filename: string;
  status: JobStatus["status"];
}

// ── Icon components (no emojis) ─────────────────────────────────────────────

function IconUpload() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}

function IconDocument() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
  );
}

function IconTrash() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6"/>
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
      <path d="M10 11v6M14 11v6"/>
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
    </svg>
  );
}

function IconCheck() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
}

function IconX() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"/>
      <line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  );
}

// ── Component ────────────────────────────────────────────────────────────────

export default function Sidebar({ documents, onDocumentsChange, stats, themeToggle }: SidebarProps) {
  const [dragOver, setDragOver] = useState(false);
  const [jobs, setJobs] = useState<UploadJob[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const pollerRef = useRef<Record<string, NodeJS.Timeout>>({});

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    for (const file of Array.from(files)) {
      try {
        const res = await uploadFile(file);
        const job: UploadJob = { job_id: res.job_id, filename: res.filename, status: "pending" };
        setJobs((prev) => [job, ...prev]);
        startPolling(res.job_id);
      } catch (e) {
        console.error("Upload error:", e);
      }
    }
  }, []);

  const startPolling = (job_id: string) => {
    const poll = async () => {
      try {
        const status = await getJobStatus(job_id);
        setJobs((prev) =>
          prev.map((j) => (j.job_id === job_id ? { ...j, status: status.status } : j))
        );
        if (status.status === "done") {
          onDocumentsChange();
          clearInterval(pollerRef.current[job_id]);
          delete pollerRef.current[job_id];
          setTimeout(() => setJobs((prev) => prev.filter((j) => j.job_id !== job_id)), 3000);
        } else if (status.status === "error") {
          clearInterval(pollerRef.current[job_id]);
          delete pollerRef.current[job_id];
        }
      } catch {
        clearInterval(pollerRef.current[job_id]);
      }
    };
    pollerRef.current[job_id] = setInterval(poll, 2000);
    poll();
  };

  const handleDelete = async (doc_id: string) => {
    setDeleting(doc_id);
    try {
      await deleteDocument(doc_id);
      onDocumentsChange();
    } catch (e) {
      console.error("Delete error:", e);
    } finally {
      setDeleting(null);
    }
  };

  const baseName = (path: string) => path.split(/[\\/]/).pop() || path;

  return (
    <aside
      style={{
        width: "268px",
        minWidth: "268px",
        background: "var(--bg-sidebar)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      {/* Brand header */}
      <div
        style={{
          padding: "16px 18px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "15px",
              fontWeight: 700,
              letterSpacing: "-0.4px",
              color: "var(--text-primary)",
            }}
          >
            Anything&apos;sOK
          </div>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "1px" }}>
            GraphRAG Assistant
          </div>
        </div>
        {themeToggle}
      </div>

      {/* Stats grid */}
      {stats && (
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid var(--border)",
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "6px",
          }}
        >
          {[
            { label: "Entities", value: stats.entities },
            { label: "Chunks", value: stats.chunks },
            { label: "Relations", value: stats.relationships },
            { label: "Communities", value: stats.communities },
          ].map((s) => (
            <div
              key={s.label}
              style={{
                background: "var(--bg-card)",
                borderRadius: "7px",
                padding: "8px 10px",
                border: "1px solid var(--border)",
                boxShadow: "var(--shadow)",
              }}
            >
              <div style={{ fontSize: "15px", fontWeight: 700, color: "var(--accent)" }}>
                {s.value.toLocaleString()}
              </div>
              <div style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "1px" }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Upload zone */}
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
        <input
          ref={fileRef}
          type="file"
          multiple
          id="file-upload-input"
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
          accept=".pdf,.txt,.md,.docx,.pptx,.csv,.xlsx,.epub,.html,.png,.jpg,.jpeg,.webp,.wav,.mp3"
        />
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
          onClick={() => fileRef.current?.click()}
          style={{
            border: `1.5px dashed ${dragOver ? "var(--accent)" : "var(--border)"}`,
            borderRadius: "8px",
            padding: "16px 12px",
            textAlign: "center",
            cursor: "pointer",
            transition: "all 0.15s ease",
            background: dragOver ? "rgba(210,143,75,0.05)" : "transparent",
            color: dragOver ? "var(--accent)" : "var(--text-muted)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "center", marginBottom: "6px" }}>
            <IconUpload />
          </div>
          <div style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-secondary)" }}>
            Upload document
          </div>
          <div style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "2px" }}>
            PDF, DOCX, TXT, images, audio
          </div>
        </div>
      </div>

      {/* Active jobs */}
      {jobs.length > 0 && (
        <div style={{ padding: "8px 16px", borderBottom: "1px solid var(--border)" }}>
          <div style={{
            fontSize: "10px", fontWeight: 600, color: "var(--text-muted)",
            textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px"
          }}>
            Processing
          </div>
          {jobs.map((job) => (
            <div
              key={job.job_id}
              className="slide-up"
              style={{
                background: "var(--bg-card)",
                border: "1px solid var(--border)",
                borderRadius: "7px",
                padding: "8px 10px",
                marginBottom: "5px",
                display: "flex",
                alignItems: "center",
                gap: "8px",
                boxShadow: "var(--shadow)",
              }}
            >
              {job.status === "done" ? (
                <span style={{ color: "#16a34a" }}><IconCheck /></span>
              ) : job.status === "error" ? (
                <span style={{ color: "#dc2626" }}><IconX /></span>
              ) : (
                <div className="spinner" />
              )}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: "11px", fontWeight: 500, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {job.filename}
                </div>
                <div style={{ fontSize: "10px", color: job.status === "error" ? "#dc2626" : "var(--text-muted)" }}>
                  {job.status === "pending" && "Queued"}
                  {job.status === "processing" && "Building knowledge graph..."}
                  {job.status === "done" && "Ingested successfully"}
                  {job.status === "error" && "Failed"}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Document list */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 16px 16px" }}>
        <div style={{
          fontSize: "10px", fontWeight: 600, color: "var(--text-muted)",
          textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "6px"
        }}>
          Documents ({documents.length})
        </div>
        {documents.length === 0 ? (
          <div style={{ color: "var(--text-muted)", fontSize: "12px", textAlign: "center", marginTop: "20px", lineHeight: "1.6" }}>
            No documents yet.<br />Upload one to get started.
          </div>
        ) : (
          documents.map((doc) => (
            <div
              key={doc.doc_id}
              className="fade-in"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "7px 9px",
                borderRadius: "7px",
                marginBottom: "3px",
                border: "1px solid transparent",
                transition: "all 0.12s ease",
                cursor: "default",
              }}
              onMouseEnter={(e) => {
                const el = e.currentTarget as HTMLDivElement;
                el.style.background = "var(--bg-card)";
                el.style.borderColor = "var(--border)";
                el.style.boxShadow = "var(--shadow)";
              }}
              onMouseLeave={(e) => {
                const el = e.currentTarget as HTMLDivElement;
                el.style.background = "transparent";
                el.style.borderColor = "transparent";
                el.style.boxShadow = "none";
              }}
            >
              <span style={{ color: "var(--text-muted)", flexShrink: 0 }}><IconDocument /></span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: "12px", fontWeight: 500, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {baseName(doc.source)}
                </div>
                <div style={{ fontSize: "10px", color: "var(--text-muted)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {doc.doc_id}
                </div>
              </div>
              <button
                onClick={() => handleDelete(doc.doc_id)}
                disabled={deleting === doc.doc_id}
                title="Delete document"
                style={{
                  background: "none", border: "none", cursor: "pointer",
                  color: "var(--text-muted)", padding: "3px", borderRadius: "4px",
                  transition: "color 0.12s", flexShrink: 0, display: "flex",
                }}
                onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.color = "#dc2626")}
                onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.color = "var(--text-muted)")}
              >
                {deleting === doc.doc_id ? <div className="spinner" style={{ width: "11px", height: "11px" }} /> : <IconTrash />}
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
