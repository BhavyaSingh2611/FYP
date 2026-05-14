<script>
  export let score = 0;
  export let mate = null;

  $: displayScore = formatScore(score, mate);
  $: whitePercent = calcPercent(score, mate);

  function formatScore(s, m) {
    if (m !== null && m !== undefined) {
      return m > 0 ? `M${m}` : `M${Math.abs(m)}`;
    }
    if (s === null || s === undefined) return "0.0";
    return (s >= 0 ? "+" : "") + s.toFixed(1);
  }

  function calcPercent(s, m) {
    if (m !== null && m !== undefined) {
      return m > 0 ? 98 : 2;
    }
    if (s === null || s === undefined) return 50;
    const clamped = Math.max(-10, Math.min(10, s));
    return 50 + 50 * (2 / (1 + Math.exp(-0.5 * clamped)) - 1);
  }
</script>

<div class="flex flex-col items-center gap-1.5 w-8 h-full">
  <div class="flex-1 w-6 rounded-sm overflow-hidden flex flex-col border border-[#333]">
    <div class="bg-[#333] transition-[height] duration-500 ease-in-out" style="height: {100 - whitePercent}%"></div>
    <div class="bg-[#e8e8e8] transition-[height] duration-500 ease-in-out" style="height: {whitePercent}%"></div>
  </div>
  <div
    class="text-[0.7rem] font-bold whitespace-nowrap font-mono"
    class:text-[#888]={score < 0 || (mate !== null && mate < 0)}
    class:text-[#e0e0e0]={!(score < 0 || (mate !== null && mate < 0))}
  >
    {displayScore}
  </div>
</div>
