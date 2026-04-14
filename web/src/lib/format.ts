export function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
}

export function formatDate(value: string | null | undefined) {
  if (!value) {
    return "Unknown";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[date.getUTCMonth()]} ${date.getUTCDate()}, ${date.getUTCFullYear()}`;
}

export function formatNumber(value: number) {
  if (Math.abs(value) >= 1000) {
    const compact = value / 1000;
    const digits = Math.abs(compact) >= 10 ? 0 : 1;
    return `${compact.toFixed(digits)} K`;
  }
  return new Intl.NumberFormat("en-US").format(value);
}
