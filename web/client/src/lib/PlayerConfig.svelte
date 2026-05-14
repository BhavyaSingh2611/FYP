<script>
    export let title = "";
    export let icon = "";
    export let runs = [];
    export let availableModels = [];
    export let stockfishLevels = [];
    export let modelLabel = (name) => name;

    export let type = "human";
    export let run = "";
    export let model = "";
    export let elo = 1500;
</script>

<div class="flex-1 bg-bg-secondary p-5 rounded-lg border border-border">
    <h2 class="text-base font-semibold text-text-primary mb-4">
        {icon}
        {title}
    </h2>
    <div class="mb-3">
        <select
            bind:value={type}
            class="w-full px-3 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary text-sm outline-none transition-colors focus:border-accent"
        >
            <option value="human">Human</option>
            <option value="model">Neural Net Model</option>
            <option value="stockfish">Stockfish</option>
        </select>
    </div>

    {#if type === "model"}
        <div class="mb-3">
            <p class="block text-xs font-medium text-text-secondary mb-1.5">
                Architecture
            </p>
            <select
                bind:value={run}
                class="w-full px-3 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary text-sm outline-none transition-colors focus:border-accent"
            >
                {#each runs as runOption}
                    <option value={runOption.name}
                        >{modelLabel(runOption.name)}</option
                    >
                {/each}
            </select>
        </div>
        <div class="mb-3">
            <p class="block text-xs font-medium text-text-secondary mb-1.5">
                Checkpoint
            </p>
            <select
                bind:value={model}
                class="w-full px-3 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary text-sm outline-none transition-colors focus:border-accent"
            >
                {#each availableModels as m}
                    <option value={m}>{m}</option>
                {/each}
            </select>
        </div>
    {/if}

    {#if type === "stockfish"}
        <div class="mb-3">
            <p class="block text-xs font-medium text-text-secondary mb-1.5">
                Skill Level (Elo)
            </p>
            <select
                bind:value={elo}
                class="w-full px-3 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary text-sm outline-none transition-colors focus:border-accent"
            >
                {#each stockfishLevels as level}
                    <option value={level}>{level}</option>
                {/each}
            </select>
        </div>
    {/if}
</div>
