<script>
  import { createEventDispatcher } from "svelte";
  import PlayerConfig from "./PlayerConfig.svelte";

  export let runs = [];
  export let loading = false;
  export let error = null;

  const dispatch = createEventDispatcher();

  const STOCKFISH_LEVELS = [1320, 1500, 1800, 2000, 2200, 2500, 3200];

  let whiteType = "human";
  let whiteRun = "";
  let whiteModel = "";
  let whiteElo = 1500;

  let blackType = "model";
  let blackRun = "";
  let blackModel = "";
  let blackElo = 1500;

  $: availableWhiteModels = runs.find((r) => r.name === whiteRun)?.models || [];
  $: if (
    availableWhiteModels.length &&
    !availableWhiteModels.includes(whiteModel)
  ) {
    whiteModel = availableWhiteModels[0];
  }

  $: availableBlackModels = runs.find((r) => r.name === blackRun)?.models || [];
  $: if (
    availableBlackModels.length &&
    !availableBlackModels.includes(blackModel)
  ) {
    blackModel = availableBlackModels[0];
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

  function getConfig(type, run, model, elo) {
    if (type === "human") return { type: "human" };
    if (type === "stockfish") return { type: "stockfish", elo };
    if (type === "model") return { type: "model", run, model };
    return { type: "human" };
  }

  function handleStart() {
    dispatch("start", {
      white: getConfig(whiteType, whiteRun, whiteModel, whiteElo),
      black: getConfig(blackType, blackRun, blackModel, blackElo),
    });
  }
</script>

<div class="flex-1 flex items-center justify-center p-8">
  <div
    class="bg-bg-card border border-border rounded-xl p-10 w-full max-w-2xl shadow-2xl"
  >
    <h1 class="text-3xl font-bold text-center text-text-primary mb-2">
      Chess ML Arena
    </h1>
    <p class="text-text-secondary text-base text-center mb-10">
      Play or spectate matches between AI models, Stockfish, and Humans
    </p>

    {#if runs.length === 0 && !error}
      <p class="text-text-secondary text-center italic">
        Loading training runs…
      </p>
    {:else if runs.length === 0 && error}
      <p class="text-danger text-sm text-center mt-4">{error}</p>
    {:else}
      <!-- Player configs with generous spacing -->
      <div class="flex gap-10 items-start mb-8 relative">
        <PlayerConfig
          title="White Player"
          icon="♔"
          {runs}
          availableModels={availableWhiteModels}
          stockfishLevels={STOCKFISH_LEVELS}
          {modelLabel}
          bind:type={whiteType}
          bind:run={whiteRun}
          bind:model={whiteModel}
          bind:elo={whiteElo}
        />

        <!-- VS badge centred in the gap -->
        <div
          class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-bg-card text-text-secondary font-bold w-10 h-10 flex items-center justify-center rounded-full border border-border text-sm z-10 shrink-0"
        >
          VS
        </div>

        <PlayerConfig
          title="Black Player"
          icon="♚"
          {runs}
          availableModels={availableBlackModels}
          stockfishLevels={STOCKFISH_LEVELS}
          {modelLabel}
          bind:type={blackType}
          bind:run={blackRun}
          bind:model={blackModel}
          bind:elo={blackElo}
        />
      </div>

      <button
        class="w-full py-3.5 border-none rounded-lg bg-accent text-white text-lg font-semibold transition-all duration-200 cursor-pointer hover:-translate-y-0.5 hover:bg-accent-hover active:translate-y-0.5 disabled:opacity-60 disabled:cursor-not-allowed"
        on:click={handleStart}
        disabled={loading}
      >
        {loading ? "Starting..." : "Start Match"}
      </button>

      {#if error}
        <p class="text-danger text-sm text-center mt-4">{error}</p>
      {/if}
    {/if}
  </div>
</div>
