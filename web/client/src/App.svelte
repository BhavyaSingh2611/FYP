<script>
  import Chessboard from "./lib/Chessboard.svelte";
  import EvalBar from "./lib/EvalBar.svelte";
  import Setup from "./lib/Setup.svelte";
  import {
    getRuns,
    newGame,
    makeMove,
    makeBotMove,
    resignGame,
  } from "./lib/api.js";
  import {
    pairMoves,
    getPlayerName,
    formatEval,
    lineScore,
    formatLineScore,
    evalClass,
  } from "./lib/utils.js";

  let runs = [];
  let game = null;
  let loading = false;
  let error = null;
  let showBestMove = false;
  let showEvalBar = true;
  let showEngineLines = true;

  $: isPlayerTurn =
    game &&
    ((game.turn === "white" && !game.white_info) ||
      (game.turn === "black" && !game.black_info) ||
      (game.turn === "white" && game.white_info.type === "human") ||
      (game.turn === "black" && game.black_info.type === "human")) &&
    game.status === "playing" &&
    !loading;

  $: gameOver = game && game.status !== "playing";
  $: pairedMoves = game ? pairMoves(game.move_history) : [];
  $: currentEval = game?.eval || null;
  $: bestMoveSquares = currentEval?.best_move
    ? {
        from: currentEval.best_move.slice(0, 2),
        to: currentEval.best_move.slice(2, 4),
      }
    : null;

  // Determine which color the human is playing so we can orient the board correctly.
  // If black is human (and white is not also human), flip the board so black is at bottom.
  $: humanColor = (() => {
    if (!game) return "white";
    const whiteIsHuman = !game.white_info || game.white_info.type === "human";
    const blackIsHuman = !game.black_info || game.black_info.type === "human";
    if (blackIsHuman && !whiteIsHuman) return "black";
    return "white"; // white-human, both-human, or bot-vs-bot → default white perspective
  })();

  // Labels: the player at bottom is always the human's color side; top is the opponent.
  $: bottomColor = humanColor;
  $: topColor = humanColor === "black" ? "white" : "black";

  // hacky fix to enable bot vs bot games, and not be stuck on the first move waiting for a bot to play
  function checkBotTurn() {
    if (game && game.status === "playing" && !error) {
      const isHuman =
        (game.turn === "white" &&
          (!game.white_info || game.white_info.type === "human")) ||
        (game.turn === "black" &&
          (!game.black_info || game.black_info.type === "human"));

      if (!isHuman) setTimeout(triggerBotMove, 50);
    }
  }

  async function triggerBotMove() {
    if (loading) return;
    loading = true;
    error = null;
    try {
      await new Promise((resolve) => setTimeout(resolve, 600));
      game = await makeBotMove(game.id);
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
    checkBotTurn();
  }

  async function init() {
    try {
      const data = await getRuns();
      runs = data.runs;
    } catch (e) {
      error = "Failed to connect to server - is it running?";
    }
  }

  async function handleSetupStart(event) {
    loading = true;
    error = null;
    try {
      game = await newGame(event.detail.white, event.detail.black);
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
    checkBotTurn();
  }

  async function handleMove(event) {
    if (!game || loading || !isPlayerTurn) return;
    loading = true;
    error = null;
    try {
      game = await makeMove(game.id, event.detail.move);
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
    checkBotTurn();
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
  <Setup {runs} {loading} {error} on:start={handleSetupStart} />
{:else}
  <div class="flex-1 flex gap-6 justify-center items-start p-7 flex-wrap">
    <div class="flex items-stretch gap-3">
      {#if showEvalBar}
        <div class="flex" style="padding-top: 44px; padding-bottom: 44px;">
          <EvalBar score={currentEval?.score} mate={currentEval?.mate} />
        </div>
      {/if}

      <div class="flex flex-col items-center gap-2">
        <div
          class="flex items-center gap-2.5 w-full px-2 py-1.5 bg-black/10 rounded-md min-h-[36px]"
        >
          <span class="text-sm font-semibold text-text-primary">
            {getPlayerName(
              topColor === "black" ? game.black_info : game.white_info,
              topColor === "black" ? "Black" : "White",
            )}
          </span>
          {#if loading && game.turn === topColor}
            <span class="thinking-dot"></span>
          {/if}
        </div>

        <Chessboard
          fen={game.fen}
          legalMoves={game.legal_moves}
          lastMove={game.last_move}
          isCheck={game.is_check}
          playerColor={humanColor}
          {isPlayerTurn}
          status={game.status}
          bestMove={showBestMove ? bestMoveSquares : null}
          on:move={handleMove}
        />

        <div
          class="flex items-center gap-2.5 w-full px-2 py-1.5 bg-black/10 rounded-md min-h-[36px]"
        >
          <span class="text-sm font-semibold text-text-primary">
            {getPlayerName(
              bottomColor === "black" ? game.black_info : game.white_info,
              bottomColor === "black" ? "Black" : "White",
            )}
          </span>
          {#if loading && game.turn === bottomColor}
            <span class="thinking-dot"></span>
          {/if}
        </div>
      </div>
    </div>

    <div
      class="mt-[44px] w-80 min-w-[280px] bg-bg-card border border-border rounded-lg flex flex-col overflow-hidden shadow-lg self-start"
    >
      <div
        class="px-4 py-4 bg-bg-secondary border-b border-border font-semibold text-[1.05rem] text-center text-text-primary"
      >
        Match Info
      </div>

      <div
        class="px-4 py-3.5 text-sm font-semibold border-b border-border text-center bg-black/10"
        class:text-danger={gameOver}
        class:text-text-primary={!gameOver}
      >
        {#if game.status === "playing"}
          {#if loading}
            <span class="text-accent">Thinking…</span>
          {:else if isPlayerTurn}
            Your turn
          {:else}
            Waiting…
          {/if}
        {:else if game.status === "checkmate"}
          Checkmate - {game.result === "1-0" ? "White" : "Black"} wins
        {:else if game.status === "stalemate"}
          Stalemate - Draw
        {:else if game.status === "draw"}
          Draw - {game.result}
        {:else if game.status === "resigned"}
          Resigned - {game.result}
        {/if}
      </div>

      <div
        class="flex items-center justify-between px-4 py-3 border-b border-border"
      >
        <span class="text-sm text-text-primary">Evaluation Bar</span>
        <label class="switch"
          ><input type="checkbox" bind:checked={showEvalBar} /><span
            class="slider"
          ></span></label
        >
      </div>
      <div
        class="flex items-center justify-between px-4 py-3 border-b border-border"
      >
        <span class="text-sm text-text-primary">Engine Lines</span>
        <label class="switch"
          ><input type="checkbox" bind:checked={showEngineLines} /><span
            class="slider"
          ></span></label
        >
      </div>
      <div
        class="flex items-center justify-between px-4 py-3 border-b border-border"
      >
        <span class="text-sm text-text-primary">Show Best Move</span>
        <label class="switch"
          ><input type="checkbox" bind:checked={showBestMove} /><span
            class="slider"
          ></span></label
        >
      </div>

      {#if showEngineLines && currentEval?.lines?.length}
        <div class="border-b border-border bg-bg-secondary">
          <div class="px-4 py-2 border-b border-border">
            <span
              class="text-[0.8rem] font-semibold text-text-secondary uppercase tracking-wide"
              >Analysis</span
            >
          </div>
          {#each currentEval.lines as line, i}
            <div
              class="flex items-center gap-3 px-4 py-1.5 text-[0.85rem] border-b border-white/[0.02] {i ===
              0
                ? 'bg-white/[0.03]'
                : ''}"
            >
              <span
                class="font-semibold min-w-[52px] {lineScore(line) > 0.3
                  ? 'eval-good'
                  : lineScore(line) < -0.3
                    ? 'eval-bad'
                    : 'eval-neutral'}"
              >
                {formatLineScore(line)}
              </span>
              <span
                class="text-text-secondary whitespace-nowrap overflow-hidden text-ellipsis"
              >
                {line.moves}
              </span>
            </div>
          {/each}
        </div>
      {/if}

      <div class="flex-1 overflow-y-auto max-h-[380px] py-2 bg-bg-card">
        {#if pairedMoves.length === 0}
          <p class="px-5 py-5 text-center text-text-secondary italic text-sm">
            No moves yet
          </p>
        {/if}
        {#each pairedMoves as pair, i}
          <div
            class="grid items-center px-4 py-1.5 text-sm"
            style="grid-template-columns: 32px 1fr 64px 1fr 64px;"
            class:bg-bg-secondary={i % 2 === 1}
          >
            <span class="text-text-secondary font-medium">{i + 1}.</span>
            <span class="font-medium text-text-primary"
              >{pair[0]?.san ?? ""}</span
            >
            <span
              class="text-[0.78rem] text-right pr-2 {evalClass(pair[0]?.eval)}"
              >{formatEval(pair[0]?.eval)}</span
            >
            <span class="font-medium text-text-primary"
              >{pair[1]?.san ?? ""}</span
            >
            <span
              class="text-[0.78rem] text-right pr-2 {evalClass(pair[1]?.eval)}"
              >{formatEval(pair[1]?.eval)}</span
            >
          </div>
        {/each}
      </div>

      <div class="flex gap-2.5 p-4 border-t border-border bg-bg-secondary">
        {#if game.status === "playing" && (game.white_info?.type === "human" || game.black_info?.type === "human")}
          <button
            class="flex-1 py-2.5 border-none rounded-lg text-sm font-semibold transition-colors bg-danger text-white hover:bg-danger-hover cursor-pointer"
            on:click={handleResign}>Resign</button
          >
        {/if}
        <button
          class="flex-1 py-2.5 rounded-lg text-sm font-semibold transition-colors bg-bg-primary text-text-primary border border-border hover:bg-bg-hover cursor-pointer"
          on:click={backToSetup}>New Match</button
        >
      </div>

      {#if error}
        <p class="text-danger text-sm px-4 pb-4 text-center">{error}</p>
      {/if}
    </div>
  </div>
{/if}
