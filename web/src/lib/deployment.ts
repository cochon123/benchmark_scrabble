export function runManagementEnabled() {
  return process.env.NODE_ENV !== "production";
}
