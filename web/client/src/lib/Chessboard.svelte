<script>
  import { createEventDispatcher } from "svelte";
  import PromotionPicker from "./PromotionPicker.svelte";

  export let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  export let legalMoves = [];
  export let lastMove = null;
  export let isCheck = false;
  export let playerColor = "white";
  export let isPlayerTurn = false;
  export let status = "playing";
  export let bestMove = null;

  const dispatch = createEventDispatcher();

  const PIECE_IMAGES = {
    K: "/pieces/wK.svg", Q: "/pieces/wQ.svg", R: "/pieces/wR.svg",
    B: "/pieces/wB.svg", N: "/pieces/wN.svg", P: "/pieces/wP.svg",
    k: "/pieces/bK.svg", q: "/pieces/bQ.svg", r: "/pieces/bR.svg",
    b: "/pieces/bB.svg", n: "/pieces/bN.svg", p: "/pieces/bP.svg",
  };

  function sqToCoords(sq) {
    const col = sq.charCodeAt(0) - 97;
    const row = 8 - parseInt(sq[1]);
    const vCol = flipped ? 7 - col : col;
    const vRow = flipped ? 7 - row : row;
    return { x: (vCol + 0.5) * 12.5, y: (vRow + 0.5) * 12.5 };
  }

  let selectedSquare = null;
  let promotionPending = null;

  $: board = parseFen(fen);
  $: flipped = playerColor === "black";
  $: legalTargets = selectedSquare ? getLegalTargets(selectedSquare) : [];
  $: kingSquare = isCheck ? findKingInCheck(board, fen) : null;

  $: if (!isPlayerTurn) selectedSquare = null;

  function parseFen(f) {
    const ranks = f.split(" ")[0].split("/");
    return ranks.map((rank) => {
      const row = [];
      for (const ch of rank) {
        if (ch >= "1" && ch <= "8") {
          for (let i = 0; i < +ch; i++) row.push(null);
        } else {
          row.push(ch);
        }
      }
      return row;
    });
  }

  function sqName(row, col) {
    return String.fromCharCode(97 + col) + (8 - row);
  }

  function getLegalTargets(sq) {
    return [...new Set(
      legalMoves.filter((m) => m.startsWith(sq)).map((m) => m.slice(2, 4))
    )];
  }

  function isOwn(row, col) {
    const p = board[row]?.[col];
    if (!p) return false;
    return playerColor === "white" ? p === p.toUpperCase() : p === p.toLowerCase();
  }

  function findKingInCheck(brd, f) {
    const turn = f.split(" ")[1];
    const king = turn === "w" ? "K" : "k";
    for (let r = 0; r < 8; r++)
      for (let c = 0; c < 8; c++)
        if (brd[r]?.[c] === king) return sqName(r, c);
    return null;
  }

  function handleClick(row, col) {
    if (status !== "playing" || !isPlayerTurn) return;

    const sq = sqName(row, col);

    if (!selectedSquare) {
      if (isOwn(row, col)) selectedSquare = sq;
      return;
    }

    if (sq === selectedSquare) {
      selectedSquare = null;
      return;
    }

    if (legalTargets.includes(sq)) {
      const promos = legalMoves.filter(
        (m) => m.startsWith(selectedSquare + sq) && m.length === 5
      );
      if (promos.length > 0) {
        promotionPending = { from: selectedSquare, to: sq };
      } else {
        dispatch("move", { move: selectedSquare + sq });
        selectedSquare = null;
      }
      return;
    }

    selectedSquare = isOwn(row, col) ? sq : null;
  }

  function handlePromotion(e) {
    if (promotionPending) {
      dispatch("move", { move: promotionPending.from + promotionPending.to + e.detail });
      promotionPending = null;
      selectedSquare = null;
    }
  }

  function cancelPromotion() {
    promotionPending = null;
    selectedSquare = null;
  }
</script>

<div class="board-wrap">
  <div class="board">
    {#each Array(8) as _, vr}
      {#each Array(8) as _, vc}
        {@const row = flipped ? 7 - vr : vr}
        {@const col = flipped ? 7 - vc : vc}
        {@const sq = sqName(row, col)}
        {@const piece = board[row]?.[col]}
        {@const dark = (row + col) % 2 === 1}
        {@const selected = sq === selectedSquare}
        {@const target = legalTargets.includes(sq)}
        {@const lastMv = lastMove && (sq === lastMove.from || sq === lastMove.to)}
        {@const inCheck = sq === kingSquare}
        <button
          class="sq"
          class:dark
          class:light={!dark}
          class:selected
          class:last-mv={lastMv && !selected}
          class:in-check={inCheck}
          on:click={() => handleClick(row, col)}
        >
          {#if piece}
            <img class="pc" src={PIECE_IMAGES[piece]} alt={piece} draggable="false" />
          {/if}

          {#if target && !piece}
            <span class="dot"></span>
          {/if}
          {#if target && piece}
            <span class="cap-ring"></span>
          {/if}

          {#if vc === 0}
            <span class="coord rank" class:coord-dark={dark} class:coord-light={!dark}>
              {8 - row}
            </span>
          {/if}
          {#if vr === 7}
            <span class="coord file" class:coord-dark={dark} class:coord-light={!dark}>
              {String.fromCharCode(97 + col)}
            </span>
          {/if}
        </button>
      {/each}
    {/each}
  </div>

  {#if bestMove}
    {@const from = sqToCoords(bestMove.from)}
    {@const to = sqToCoords(bestMove.to)}
    <svg class="arrow-overlay" viewBox="0 0 100 100">
      <defs>
        <marker id="arrowhead" markerWidth="4" markerHeight="3" refX="3.5" refY="1.5" orient="auto">
          <polygon points="0 0, 4 1.5, 0 3" fill="rgba(129,182,76,0.8)" />
        </marker>
      </defs>
      <line x1={from.x} y1={from.y} x2={to.x} y2={to.y}
        stroke="rgba(129,182,76,0.7)" stroke-width="2.5" stroke-linecap="round"
        marker-end="url(#arrowhead)" />
    </svg>
  {/if}

  {#if promotionPending}
    <PromotionPicker
      color={playerColor}
      on:select={handlePromotion}
      on:cancel={cancelPromotion}
    />
  {/if}
</div>

<style>
  .board-wrap {
    position: relative;
    width: min(560px, 88vw);
    aspect-ratio: 1;
  }
  .board {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    width: 100%;
    height: 100%;
    border-radius: 4px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
  }

  .sq {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    padding: 0;
    margin: 0;
    cursor: pointer;
    outline: none;
    aspect-ratio: 1;
  }
  .sq:focus-visible {
    outline: 2px solid #fff;
    outline-offset: -2px;
  }
  .light            { background: #f0d9b5; }
  .dark             { background: #b58863; }
  .light.last-mv    { background: #f7f769; }
  .dark.last-mv     { background: #baca2b; }
  .light.selected   { background: #b4d98c; }
  .dark.selected    { background: #829769; }
  .in-check         { background: radial-gradient(ellipse at center,
                        rgba(255,0,0,.75) 0%, rgba(200,0,0,.35) 60%, transparent 100%) !important; }

  .pc {
    width: 85%;
    height: 85%;
    pointer-events: none;
    user-select: none;
    filter: drop-shadow(0 1px 2px rgba(0,0,0,0.4));
  }
  .arrow-overlay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 10;
  }

  .dot {
    width: 28%;
    height: 28%;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.18);
    pointer-events: none;
  }
  .cap-ring {
    position: absolute;
    inset: 4%;
    border-radius: 50%;
    border: 4.5px solid rgba(0, 0, 0, 0.18);
    pointer-events: none;
  }

  .coord {
    position: absolute;
    font-size: 0.65rem;
    font-weight: 700;
    pointer-events: none;
    user-select: none;
  }
  .rank { top: 2px; left: 3px; }
  .file { bottom: 1px; right: 3px; }
  .coord-dark  { color: #f0d9b5; }
  .coord-light { color: #b58863; }
</style>
