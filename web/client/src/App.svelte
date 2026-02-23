<script>
  import Chessboard from "./lib/Chessboard.svelte";
  import EvalBar from "./lib/EvalBar.svelte";
  import { getRuns, newGame, makeMove, resignGame } from "./lib/api.js";

  let runs = [];
  let selectedRun = "";
  let selectedModel = "";
  let playerColor = "white";
  let game = null;
  let loading = false;
  let error = null;
  let showBestMove = false;
  let showEvalBar = true;
  let showEngineLines = true;

  $: availableModels = runs.find((r) => r.name === selectedRun)?.models || [];
  $: if (availableModels.length && !availableModels.includes(selectedModel)) {
    selectedModel = availableModels[0];
  }
  $: isPlayerTurn =
    game &&
    game.turn === game.player_color &&
    game.status === "playing" &&
    !loading;
  $: gameOver = game && game.status !== "playing";
  $: pairedMoves = game ? pairMoves(game.move_history) : [];
  $: currentEval = game?.eval || null;
  $: bestMoveSquares = currentEval?.best_move
    ? { from: currentEval.best_move.slice(0, 2), to: currentEval.best_move.slice(2, 4) }
    : null;

  function pairMoves(history) {
    const pairs = [];
    for (let i = 0; i < history.length; i += 2) {
      pairs.push([history[i], history[i + 1] || null]);
    }
    return pairs;
  }

  function modelLabel(name) {
    const labels = {
      convnet: "ConvNet",
      resnet: "ResNet",
      square_transformer: "Square Transformer",
      piece_transformer: "Piece Transformer",
      gcn: "GCN",
      gat: "GAT",
    };
    return labels[name] || name;
  }

  function formatEval(ev) {
    if (!ev) return "";
    if (ev.mate !== null && ev.mate !== undefined) return ev.mate > 0 ? `M${ev.mate}` : `M${Math.abs(ev.mate)}`;
    if (ev.score === null || ev.score === undefined) return "";
    return (ev.score >= 0 ? "+" : "") + ev.score.toFixed(1);
  }

  function lineScore(line) {
    if (line.mate !== null && line.mate !== undefined) return line.mate > 0 ? 999 : -999;
    return line.score ?? 0;
  }

  function formatLineScore(line) {
    if (line.mate !== null && line.mate !== undefined) return line.mate > 0 ? `M${line.mate}` : `M${Math.abs(line.mate)}`;
    if (line.score === null || line.score === undefined) return "0.0";
    return (line.score >= 0 ? "+" : "") + line.score.toFixed(2);
  }

  function evalClass(ev) {
    if (!ev) return "eval-neutral";
    if (ev.mate !== null && ev.mate !== undefined) return ev.mate > 0 ? "eval-good" : "eval-bad";
    if (ev.score > 0.3) return "eval-good";
    if (ev.score < -0.3) return "eval-bad";
    return "eval-neutral";
  }

  async function init() {
    try {
      const data = await getRuns();
      runs = data.runs;
      if (runs.length > 0) {
        selectedRun = runs[0].name;
        if (runs[0].models.length > 0) {
          selectedModel = runs[0].models[0];
        }
      }
    } catch (e) {
      error = "Failed to connect to server — is it running?";
    }
  }

  async function startGame() {
    if (!selectedModel || !selectedRun) return;
    loading = true;
    error = null;
    try {
      game = await newGame(selectedModel, selectedRun, playerColor);
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  async function handleMove(event) {
    if (!game || loading) return;
    loading = true;
    error = null;
    try {
      game = await makeMove(game.id, event.detail.move);
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  async function handleResign() {
    if (!game) return;
    try {
      game = await resignGame(game.id);
    } catch (e) {
      error = e.message;
    }
  }

  function backToSetup() {
    game = null;
    error = null;
    showBestMove = false;
  }

  init();
</script>

{#if !game}
  <!-- ===== SETUP SCREEN ===== -->
  <div class="setup">
    <div class="setup-card">
      <h1>♔ Chess ML Arena</h1>
      <p class="subtitle">Play against your trained neural network models</p>

      {#if runs.length === 0 && !error}
        <p class="hint">Loading runs…</p>
      {:else if runs.length === 0 && error}
        <p class="error-msg">{error}</p>
      {:else}
        <div class="field">
          <label for="run-select">Training Run</label>
          <select id="run-select" bind:value={selectedRun}>
            {#each runs as run}
              <option value={run.name}>{run.name}</option>
            {/each}
          </select>
        </div>

        <div class="field">
          <label for="model-select">Model</label>
          <select id="model-select" bind:value={selectedModel}>
            {#each availableModels as m}
              <option value={m}>{modelLabel(m)}</option>
            {/each}
          </select>
        </div>

        <div class="field">
          <label>Play as</label>
          <div class="color-toggle">
            <button
              class:active={playerColor === "white"}
              on:click={() => (playerColor = "white")}
            >
              ♔ White
            </button>
            <button
              class:active={playerColor === "black"}
              on:click={() => (playerColor = "black")}
            >
              ♚ Black
            </button>
          </div>
        </div>

        <button
          class="start-btn"
          on:click={startGame}
          disabled={!selectedModel || loading}
        >
          {loading ? "Loading model…" : "Start Game"}
        </button>

        {#if error}
          <p class="error-msg">{error}</p>
        {/if}
      {/if}
    </div>
  </div>
{:else}
  <!-- ===== GAME SCREEN ===== -->
  <div class="game">
    <div class="board-area">
      {#if showEvalBar}
        <div class="eval-col">
          <EvalBar score={currentEval?.score} mate={currentEval?.mate} />
        </div>
      {/if}
      <div class="board-col">
        <div class="player-info top">
          <span class="player-name">{modelLabel(game.model_name)}</span>
          {#if loading && game.turn !== game.player_color}
            <span class="thinking-dot"></span>
          {/if}
        </div>
        <Chessboard
          fen={game.fen}
          legalMoves={game.legal_moves}
          lastMove={game.last_move}
          isCheck={game.is_check}
          playerColor={game.player_color}
          {isPlayerTurn}
          status={game.status}
          bestMove={showBestMove ? bestMoveSquares : null}
          on:move={handleMove}
        />
        <div class="player-info bottom">
          <span class="player-name">You</span>
        </div>
      </div>
    </div>

    <div class="sidebar">
      <div class="sidebar-header">
        <div class="model-badge">{modelLabel(game.model_name)}</div>
        <div class="run-badge">{game.run_name}</div>
        <div class="sf-status">
          <span class="sf-dot"></span>
          <span class="sf-label">Stockfish</span>
        </div>
      </div>

      <div class="status-bar" class:status-over={gameOver}>
        {#if game.status === "playing"}
          {#if loading}
            AI is thinking…
          {:else if isPlayerTurn}
            Your turn
          {:else}
            Waiting…
          {/if}
        {:else if game.status === "checkmate"}
          Checkmate — {game.result === "1-0" ? "White" : "Black"} wins
        {:else if game.status === "stalemate"}
          Stalemate — Draw
        {:else if game.status === "draw"}
          Draw — {game.result}
        {:else if game.status === "resigned"}
          Resigned — {game.result}
        {/if}
      </div>

      <div class="toggle-row">
        <span class="toggle-label">Eval Bar</span>
        <button class="toggle-btn" class:toggle-on={showEvalBar} on:click={() => showEvalBar = !showEvalBar}>
          {showEvalBar ? "ON" : "OFF"}
        </button>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">Engine Lines</span>
        <button class="toggle-btn" class:toggle-on={showEngineLines} on:click={() => showEngineLines = !showEngineLines}>
          {showEngineLines ? "ON" : "OFF"}
        </button>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">Best Move</span>
        <button class="toggle-btn" class:toggle-on={showBestMove} on:click={() => showBestMove = !showBestMove}>
          {showBestMove ? "ON" : "OFF"}
        </button>
      </div>

      {#if showEngineLines && currentEval?.lines?.length}
        <div class="engine-lines">
          <div class="engine-header">
            <span class="engine-title">ENGINE LINES</span>
          </div>
          {#each currentEval.lines as line, i}
            <div class="engine-line" class:engine-line-top={i === 0}>
              <span class="line-score" class:eval-good={lineScore(line) > 0.3} class:eval-bad={lineScore(line) < -0.3} class:eval-neutral={Math.abs(lineScore(line)) <= 0.3}>
                {formatLineScore(line)}
              </span>
              <span class="line-moves">{line.moves}</span>
            </div>
          {/each}
        </div>
      {/if}

      <div class="moves-panel">
        {#if pairedMoves.length === 0}
          <p class="moves-empty">No moves yet</p>
        {/if}
        {#each pairedMoves as pair, i}
          <div class="move-row" class:move-row-alt={i % 2 === 1}>
            <span class="move-num">{i + 1}.</span>
            <span class="move-san">{pair[0]?.san ?? ""}</span>
            <span class="move-eval {evalClass(pair[0]?.eval)}">{formatEval(pair[0]?.eval)}</span>
            <span class="move-san">{pair[1]?.san ?? ""}</span>
            <span class="move-eval {evalClass(pair[1]?.eval)}">{formatEval(pair[1]?.eval)}</span>
          </div>
        {/each}
      </div>

      <div class="controls">
        {#if game.status === "playing"}
          <button class="btn btn-danger" on:click={handleResign}>Resign</button>
        {/if}
        <button class="btn btn-secondary" on:click={backToSetup}>New Game</button>
      </div>

      {#if error}
        <p class="error-msg sidebar-err">{error}</p>
      {/if}
    </div>
  </div>
{/if}

<style>
  /* ---- SETUP ---- */
  .setup {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  .setup-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 40px 36px;
    width: 100%;
    max-width: 400px;
    text-align: center;
  }
  .setup-card h1 {
    font-size: 1.6rem;
    margin-bottom: 6px;
  }
  .subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 28px;
  }
  .hint {
    color: var(--text-secondary);
    font-size: 0.9rem;
  }

  .field {
    margin-bottom: 18px;
    text-align: left;
  }
  .field label {
    display: block;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }
  .field select {
    width: 100%;
    padding: 10px 12px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-primary);
    font-size: 0.95rem;
    outline: none;
  }
  .field select:focus {
    border-color: var(--accent);
  }

  .color-toggle {
    display: flex;
    gap: 0;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .color-toggle button {
    flex: 1;
    padding: 10px;
    border: none;
    background: var(--bg-card);
    color: var(--text-secondary);
    font-size: 0.95rem;
    transition: background 0.15s, color 0.15s;
  }
  .color-toggle button:first-child {
    border-right: 1px solid var(--border);
  }
  .color-toggle button.active {
    background: var(--accent);
    color: #fff;
    font-weight: 600;
  }

  .start-btn {
    width: 100%;
    padding: 12px;
    margin-top: 8px;
    border: none;
    border-radius: var(--radius);
    background: var(--accent);
    color: #fff;
    font-size: 1rem;
    font-weight: 600;
    transition: background 0.15s;
  }
  .start-btn:hover:not(:disabled) {
    background: var(--accent-hover);
  }
  .start-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* ---- GAME ---- */
  .game {
    flex: 1;
    display: flex;
    gap: 20px;
    justify-content: center;
    align-items: flex-start;
    padding: 24px;
    flex-wrap: wrap;
  }
  .board-area {
    display: flex;
    align-items: stretch;
    gap: 8px;
  }
  .eval-col {
    display: flex;
    padding: 28px 0;
  }
  .board-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }

  .player-info {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 4px 2px;
  }
  .player-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
  }
  .thinking-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .sidebar {
    width: 300px;
    min-width: 260px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-header {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .model-badge {
    background: var(--accent);
    color: #fff;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
  }
  .run-badge {
    font-size: 0.78rem;
    color: var(--text-secondary);
    background: var(--bg-card);
    padding: 3px 10px;
    border-radius: 4px;
  }
  .sf-status {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-left: auto;
  }
  .sf-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
  }
  .sf-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-weight: 600;
  }

  .status-bar {
    padding: 10px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
  }
  .status-over {
    color: var(--danger);
  }

  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
  }
  .toggle-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .toggle-btn {
    padding: 4px 14px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-secondary);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    transition: all 0.15s;
  }
  .toggle-on {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }

  .engine-lines {
    border-bottom: 1px solid var(--border);
  }
  .engine-header {
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
  }
  .engine-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
  }
  .engine-line {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 16px;
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.03);
  }
  .engine-line-top {
    background: rgba(255,255,255,0.03);
  }
  .line-score {
    font-weight: 700;
    min-width: 48px;
    font-size: 0.78rem;
  }
  .line-moves {
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .moves-panel {
    flex: 1;
    overflow-y: auto;
    max-height: 420px;
    padding: 4px 0;
  }
  .moves-empty {
    padding: 16px;
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.85rem;
  }
  .move-row {
    display: grid;
    grid-template-columns: 32px 1fr 44px 1fr 44px;
    padding: 5px 12px;
    font-size: 0.88rem;
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    align-items: center;
  }
  .move-row-alt {
    background: rgba(255, 255, 255, 0.025);
  }
  .move-num {
    color: var(--text-secondary);
  }
  .move-san {
    padding-left: 4px;
  }
  .move-eval {
    font-size: 0.72rem;
    text-align: right;
    padding-right: 4px;
  }
  :global(.eval-good) { color: #81b64c; }
  :global(.eval-bad) { color: #c33; }
  :global(.eval-neutral) { color: #888; }

  .controls {
    display: flex;
    gap: 8px;
    padding: 12px;
    border-top: 1px solid var(--border);
  }
  .btn {
    flex: 1;
    padding: 9px 0;
    border: none;
    border-radius: var(--radius);
    font-size: 0.85rem;
    font-weight: 600;
    transition: background 0.15s;
  }
  .btn-danger {
    background: var(--danger);
    color: #fff;
  }
  .btn-danger:hover {
    background: var(--danger-hover);
  }
  .btn-secondary {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover {
    background: var(--bg-hover);
  }

  .error-msg {
    color: var(--danger);
    font-size: 0.85rem;
    margin-top: 14px;
  }
  .sidebar-err {
    padding: 0 12px 12px;
    margin-top: 0;
  }
</style>
