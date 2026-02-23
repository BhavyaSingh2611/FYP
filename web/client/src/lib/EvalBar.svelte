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

<div class="eval-bar">
  <div class="bar-track">
    <div class="bar-black" style="height: {100 - whitePercent}%"></div>
    <div class="bar-white" style="height: {whitePercent}%"></div>
  </div>
  <div class="eval-label" class:eval-black={score < 0 || (mate !== null && mate < 0)}>
    {displayScore}
  </div>
</div>

<style>
  .eval-bar {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    width: 32px;
    height: 100%;
  }
  .bar-track {
    flex: 1;
    width: 24px;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    border: 1px solid #333;
  }
  .bar-black {
    background: #333;
    transition: height 0.5s ease;
  }
  .bar-white {
    background: #e8e8e8;
    transition: height 0.5s ease;
  }
  .eval-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #e0e0e0;
    font-family: "SF Mono", "Menlo", monospace;
    white-space: nowrap;
  }
  .eval-black {
    color: #888;
  }
</style>
