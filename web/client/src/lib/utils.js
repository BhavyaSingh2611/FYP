export function pairMoves(history = []) {
  const pairs = [];

  for (let i = 0; i < history.length; i += 2) {
    pairs.push([history[i], history[i + 1] || null]);
  }

  return pairs;
}

export function getPlayerName(info, fallback) {
  if (!info) return fallback;

  if (info.type === "human") return "Human Player";
  if (info.type === "stockfish") return `Stockfish (${info.elo})`;
  if (info.type === "model") return `${info.model}`;

  return fallback;
}

export function formatEval(ev) {
  if (!ev) return "";

  if (ev.mate !== null && ev.mate !== undefined) {
    return ev.mate > 0 ? `M${ev.mate}` : `M${Math.abs(ev.mate)}`;
  }

  if (ev.score === null || ev.score === undefined) return "";

  return (ev.score >= 0 ? "+" : "") + ev.score.toFixed(1);
}

export function lineScore(line) {
  if (line?.mate !== null && line?.mate !== undefined) {
    return line.mate > 0 ? 999 : -999;
  }

  return line?.score ?? 0;
}

export function formatLineScore(line) {
  if (line?.mate !== null && line?.mate !== undefined) {
    return line.mate > 0 ? `M${line.mate}` : `M${Math.abs(line.mate)}`;
  }

  if (line?.score === null || line?.score === undefined) return "0.0";

  return (line.score >= 0 ? "+" : "") + line.score.toFixed(2);
}

export function evalClass(ev) {
  if (!ev) return "eval-neutral";
  
  if (ev.mate !== null && ev.mate !== undefined) {
    return ev.mate > 0 ? "eval-good" : "eval-bad";
  }

  if (ev.score > 0.3) return "eval-good";
  if (ev.score < -0.3) return "eval-bad";

  return "eval-neutral";
}