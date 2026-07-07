type BackendStatus = { url: string; ready: boolean };

let backendStatus: BackendStatus | null = null;

export async function getBackendUrl(): Promise<string> {
  if (backendStatus) return backendStatus.url;

  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const url = await invoke<string>("get_backend_url");
    backendStatus = { url, ready: true };
    return url;
  } catch {
    const fallback = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
    backendStatus = { url: fallback, ready: false };
    return fallback;
  }
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const base = await getBackendUrl();
  const res = await fetch(`${base}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json();
}
