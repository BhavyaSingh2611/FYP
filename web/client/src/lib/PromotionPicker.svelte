<script>
  import { createEventDispatcher } from "svelte";

  export let color = "white";

  const dispatch = createEventDispatcher();

  const options = {
    white: [
      { piece: "q", img: "/pieces/wQ.svg", name: "Queen" },
      { piece: "r", img: "/pieces/wR.svg", name: "Rook" },
      { piece: "b", img: "/pieces/wB.svg", name: "Bishop" },
      { piece: "n", img: "/pieces/wN.svg", name: "Knight" },
    ],
    black: [
      { piece: "q", img: "/pieces/bQ.svg", name: "Queen" },
      { piece: "r", img: "/pieces/bR.svg", name: "Rook" },
      { piece: "b", img: "/pieces/bB.svg", name: "Bishop" },
      { piece: "n", img: "/pieces/bN.svg", name: "Knight" },
    ],
  };

  $: pieces = options[color] || options.white;
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="overlay" on:click={() => dispatch("cancel")}>
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="picker" on:click|stopPropagation>
    <p>Promote to</p>
    <div class="options">
      {#each pieces as { piece, img, name }}
        <button on:click={() => dispatch("select", piece)} title={name}>
          <img src={img} alt={name} draggable="false" class="promo-img" />
        </button>
      {/each}
    </div>
  </div>
</div>

<style>
  .overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 20;
    border-radius: 4px;
  }
  .promo-img {
    width: 80%;
    height: 80%;
    pointer-events: none;
  }
  .picker {
    background: var(--bg-card, #252525);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  }
  .picker p {
    margin-bottom: 12px;
    font-size: 0.9rem;
    color: #bababa;
  }
  .options {
    display: flex;
    gap: 8px;
  }
  .options button {
    width: 56px;
    height: 56px;
    font-size: 2.2rem;
    background: var(--bg-secondary, #1a1a1a);
    border: 2px solid var(--border, #333);
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.15s, background 0.15s;
    line-height: 1;
  }
  .options button:hover {
    border-color: var(--accent, #81b64c);
    background: var(--bg-hover, #333);
  }
</style>
