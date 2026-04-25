const BASE = "";

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export function getRuns() {
  return fetchJson(`${BASE}/api/runs`);
}

export function newGame(whiteConfig, blackConfig) {
  return fetchJson(`${BASE}/api/game/new`, {
    method: "POST",
    body: JSON.stringify({ white: whiteConfig, black: blackConfig }),
  });
}

export function makeMove(gameId, move) {
  return fetchJson(`${BASE}/api/game/${gameId}/move`, {
    method: "POST",
    body: JSON.stringify({ move }),
  });
}

export function makeBotMove(gameId) {
  return fetchJson(`${BASE}/api/game/${gameId}/bot_move`, {
    method: "POST",
  });
}

export function resignGame(gameId) {
  return fetchJson(`${BASE}/api/game/${gameId}/resign`, { method: "POST" });
}

export function getEval(gameId) {
  return fetchJson(`${BASE}/api/game/${gameId}/eval`);
}

