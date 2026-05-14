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
<div
  class="absolute inset-0 bg-black/60 flex items-center justify-center z-20 rounded-sm"
  on:click={() => dispatch("cancel")}
>
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div
    class="bg-bg-card rounded-lg px-5 py-4 text-center shadow-[0_8px_32px_rgba(0,0,0,0.5)]"
    on:click|stopPropagation
  >
    <p class="mb-3 text-[0.9rem] text-[#bababa]">Promote to</p>
    <div class="flex gap-2">
      {#each pieces as { piece, img, name }}
        <button
          class="w-14 h-14 bg-bg-secondary border-2 border-border rounded-md cursor-pointer flex items-center justify-center transition-colors duration-150 hover:border-accent hover:bg-bg-hover"
          on:click={() => dispatch("select", piece)}
          title={name}
        >
          <img src={img} alt={name} draggable="false" class="w-4/5 h-4/5 pointer-events-none" />
        </button>
      {/each}
    </div>
  </div>
</div>
