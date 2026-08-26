#!/usr/bin/env node
"use strict";

// Thin resolver, not a runtime. waypost is a Python package; this script's
// only job is finding a Python entry point that can run it and exec'ing
// straight through, argv and exit code untouched. No bundled interpreter:
// tree-sitter (C) is the hot path, so shipping a second language runtime
// here would not make anything faster, only heavier to install.
//
// Resolution order, first hit wins:
//   1. `waypost` already on PATH (installed via pip/pipx/uv tool globally)
//   2. `uvx waypost` (uv's ephemeral-environment runner)
//   3. `pipx run waypost`
//   4. `python3 -m waypost` / `python -m waypost` (requires a prior
//      `pip install waypost` into that interpreter)
//
// No network call is made here beyond what the chosen backend makes on its
// own (uvx/pipx resolve the package from PyPI on first run, same as `npx`
// itself does) -- the indexer these hand off to still makes none.

const { spawnSync } = require("node:child_process");
const { execFileSync } = require("node:child_process");

const args = process.argv.slice(2);

function commandExists(cmd) {
  const checker = process.platform === "win32" ? "where" : "which";
  try {
    execFileSync(checker, [cmd], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function run(cmd, cmdArgs) {
  const result = spawnSync(cmd, cmdArgs, { stdio: "inherit" });
  if (result.error) {
    return null; // couldn't spawn at all -- try the next candidate
  }
  return result.status === null ? 1 : result.status;
}

const candidates = [];
if (commandExists("waypost")) {
  candidates.push(["waypost", args]);
}
if (commandExists("uvx")) {
  candidates.push(["uvx", ["waypost", ...args]]);
}
if (commandExists("pipx")) {
  candidates.push(["pipx", ["run", "waypost", ...args]]);
}
for (const py of ["python3", "python"]) {
  if (commandExists(py)) {
    candidates.push([py, ["-m", "waypost", ...args]]);
  }
}

if (candidates.length === 0) {
  process.stderr.write(
    "waypost: no Python found. Install Python 3.10+, or install `uv` " +
      "(https://docs.astral.sh/uv/) or `pipx` and re-run.\n"
  );
  process.exit(2);
}

for (const [cmd, cmdArgs] of candidates) {
  const status = run(cmd, cmdArgs);
  if (status !== null) {
    process.exit(status);
  }
  // spawn failed even though the checker found it on PATH (e.g. race,
  // permissions) -- fall through and try the next candidate.
}

process.stderr.write("waypost: found a Python launcher but could not run it.\n");
process.exit(2);
