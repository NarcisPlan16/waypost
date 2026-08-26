#!/usr/bin/env node
"use strict";

// Thin resolver, not a runtime. waypost is a Python package; this script's
// only job is finding a backend that can run it and handing the invocation
// over, argv and exit code untouched. No bundled interpreter: tree-sitter
// (C) is the hot path, so shipping a second language runtime here would not
// make anything faster, only heavier to install.
//
// Resolution order, first *working* one wins:
//   1. `waypost` already on PATH (installed via pip/pipx/uv tool globally)
//   2. `uvx waypost` (uv's ephemeral-environment runner)
//   3. `pipx run waypost`
//   4. `python3 -m waypost` / `python -m waypost` (requires a prior
//      `pip install waypost` into that interpreter)
//
// "First *working* one" is the load-bearing word. An earlier version took
// the first candidate that merely existed on PATH and handed it the user's
// arguments, which broke on a plain Windows box: `python3` there is usually
// the Microsoft Store's app-execution alias, a real file that resolves fine
// and then exits 9009 with "Python was not found", swallowing the whole
// invocation while a working `python` sat further down PATH. So each
// candidate is probed with `--version` first and only used if it answers as
// waypost. That costs one extra process on the winning candidate; for uvx
// and pipx the probe also warms their cache, so the real run right after it
// is the fast one.
//
// No network call is made here beyond what the chosen backend makes on its
// own (uvx/pipx resolve the package from PyPI on first run, same as `npx`
// itself does) -- the indexer these hand off to still makes none.

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const IS_WINDOWS = process.platform === "win32";
const args = process.argv.slice(2);

// PATH is scanned here rather than shelled out to `which`/`where`: it is one
// less process per candidate, it works the same on a box where `which` is a
// shell builtin only, and it hands back the extension, which the .cmd/.bat
// case below needs.
const PATH_EXTS = IS_WINDOWS
  ? (process.env.PATHEXT || ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean)
  : [""];

function isExecutableFile(candidate) {
  try {
    if (!fs.statSync(candidate).isFile()) {
      return false;
    }
    fs.accessSync(candidate, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/** Absolute path of `cmd` on PATH, or null. */
function resolveOnPath(cmd) {
  const dirs = (process.env.PATH || "").split(path.delimiter).filter(Boolean);
  for (const dir of dirs) {
    for (const ext of PATH_EXTS) {
      const candidate = path.join(dir, cmd + ext);
      if (isExecutableFile(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

/**
 * Quote one argument the way CommandLineToArgvW un-quotes it: double every
 * run of backslashes that precedes a quote (or the closing quote), and
 * escape embedded quotes.
 */
function quoteForCmd(arg) {
  const escaped = arg.replace(/(\\*)"/g, '$1$1\\"').replace(/(\\*)$/, "$1$1");
  return `"${escaped}"`;
}

/**
 * Spawn a resolved executable. Windows shims installed as .cmd/.bat (scoop,
 * some npm-adjacent installs) cannot be spawned directly -- Node refuses
 * since the CVE-2024-27980 fix -- so those go through COMSPEC explicitly
 * rather than via `shell: true`, whose naive argument join would mangle a
 * glob or a path with spaces.
 */
function spawn(exe, exeArgs, stdio) {
  if (IS_WINDOWS && /\.(cmd|bat)$/i.test(exe)) {
    const line = [exe, ...exeArgs].map(quoteForCmd).join(" ");
    return spawnSync(process.env.COMSPEC || "cmd.exe", ["/d", "/s", "/c", `"${line}"`], {
      stdio,
      windowsVerbatimArguments: true,
    });
  }
  return spawnSync(exe, exeArgs, { stdio });
}

/** Does this candidate actually run waypost? */
function answersAsWaypost(exe, prefix) {
  const result = spawn(exe, [...prefix, "--version"], ["ignore", "pipe", "pipe"]);
  return !result.error && result.status === 0 && /waypost/i.test(String(result.stdout));
}

const candidates = [];
function consider(cmd, prefix, label) {
  const exe = resolveOnPath(cmd);
  if (exe !== null) {
    candidates.push({ exe, prefix, label });
  }
}

consider("waypost", [], "waypost");
consider("uvx", ["waypost"], "uvx waypost");
consider("pipx", ["run", "waypost"], "pipx run waypost");
for (const py of ["python3", "python"]) {
  consider(py, ["-m", "waypost"], `${py} -m waypost`);
}

if (candidates.length === 0) {
  process.stderr.write(
    "waypost: no Python launcher found on PATH. Install Python 3.10+ and " +
      "`pip install waypost`, or install `uv` (https://docs.astral.sh/uv/) " +
      "or `pipx` and re-run.\n"
  );
  process.exit(2);
}

for (const { exe, prefix, label } of candidates) {
  if (!answersAsWaypost(exe, prefix)) {
    continue;
  }
  const result = spawn(exe, [...prefix, ...args], "inherit");
  if (result.error) {
    continue; // raced with the probe (uninstalled mid-run, permissions) -- try the next
  }
  process.exit(result.status === null ? 1 : result.status);
}

process.stderr.write(
  "waypost: found launchers on PATH but none of them could run waypost.\n" +
    `Tried: ${candidates.map((c) => c.label).join(", ")}.\n` +
    "Install it with `pip install waypost`, or install `uv` " +
    "(https://docs.astral.sh/uv/) so `uvx waypost` can fetch it.\n"
);
process.exit(2);
