// lib/api.ts — typed helpers for the FastAPI backend

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────

export interface Document {
  doc_id: string;
  source: string;
}

export interface GraphStats {
  entities: number;
  chunks: number;
  relationships: number;
  communities: number;
}

export interface UploadResponse {
  job_id: string;
  filename: string;
  status: "pending" | "processing" | "done" | "error";
}

export interface JobStatus {
  status: "pending" | "processing" | "done" | "error";
  filename?: string;
  doc_id?: string | null;
  error?: string | null;
}

// ── Helpers ──────────────────────────────────────────────────────────

export async function fetchDocuments(): Promise<Document[]> {
  const res = await fetch(`${API_BASE}/api/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  const data = await res.json();
  return data.documents as Document[];
}

export async function fetchStats(): Promise<GraphStats> {
  const res = await fetch(`${API_BASE}/api/stats`);
  if (!res.ok) throw new Error("Failed to fetch stats");
  return res.json();
}

export async function deleteDocument(doc_id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(doc_id)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to delete document");
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}

export async function getJobStatus(job_id: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/upload/status/${job_id}`);
  if (!res.ok) throw new Error("Failed to get job status");
  return res.json();
}

/**
 * Stream a chat response. Calls onToken for each SSE token and onDone when finished.
 */
export async function streamChat(
  query: string,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: string) => void,
  model: string = "qwen3-small-ctx"
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, model }),
  });

  if (!res.ok || !res.body) {
    onError("Chat request failed");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split("\n");

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") {
          onDone();
          return;
        }
        if (data.startsWith("[ERROR]")) {
          onError(data.slice(7).trim());
          return;
        }
        // Restore escaped newlines
        onToken(data.replace(/\\n/g, "\n"));
      }
    }
  }
  onDone();
}
