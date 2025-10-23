export function botColor(id: number | string): string {
  const colors = ["#b30000", "#7c1158", "#4421af", "#1a53ff", "#0d88e6", "#00b7c7", "#5ad45a", "#8be04e", "#ebdc78", "#ff9f1a", "#ff6f61", "#ff4f9e"];

  // check if id is string
  if (typeof id === "string") {
    id = parseInt(id.replace(/\D/g, ''))
  }
  return colors[id % colors.length];
  };