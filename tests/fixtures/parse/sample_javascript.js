/**
 * Fixture: a plausible JavaScript module.
 *
 * Includes both module systems on purpose -- a real repository mixes
 * `require` and `import`, and the reference graph has to see both.
 */

const path = require("node:path");
import { Logger } from "./logger";

export const DEFAULT_TIMEOUT = 30;
const INTERNAL_RETRIES = 3;

/** Runtime configuration. */
export class Config {
  constructor(timeout) {
    this.timeout = timeout;
  }

  /** Serialise to a plain object. */
  toObject() {
    return { timeout: this.timeout };
  }

  static fromEnv() {
    return new Config(DEFAULT_TIMEOUT);
  }

  get isDefault() {
    return this.timeout === DEFAULT_TIMEOUT;
  }
}

/** HTTP client with retry and pooling. */
export class Client {
  constructor(log) {
    this.log = log;
  }

  /** Issue a request, retrying on transient failures. */
  send(body) {
    return this.retry(body);
  }

  retry(body) {
    return path.join(body);
  }

  close() {
    this.log.info("closed");
  }
}

/** Build a client with sensible defaults. */
export function buildClient(log) {
  return new Client(log);
}

export function describe(shape) {
  return `${shape.area()}`;
}

function normalise(url) {
  return url.trim();
}

export const parseUrl = (raw) => normalise(raw);

export const retryFor = function retryLoop(n) {
  return n * INTERNAL_RETRIES;
};

export class Pool {
  constructor(size) {
    this.size = size;
    this.free = [];
  }

  acquire() {
    return this.free.pop();
  }

  release(conn) {
    this.free.push(conn);
  }

  get available() {
    return this.free.length;
  }
}

export function withDefaults(opts) {
  return { retries: INTERNAL_RETRIES, ...opts };
}

function* iterate(items) {
  yield* items;
}

export const SCHEMA_VERSION = 1;

const registry = new Map();

export function register(name, factory) {
  registry.set(name, factory);
}

export function resolve(name) {
  return registry.get(name);
}

const handlers = {
  onOpen: () => Logger.info("open"),
  onClose: function () {
    return null;
  },
};

// A feature-detection guard. `FALLBACK_TRANSPORT` is scoped to the block, not
// to the module, and is nobody's navigation target.
if (typeof globalThis.fetch === "undefined") {
  const FALLBACK_TRANSPORT = "xhr";
  registry.set("transport", FALLBACK_TRANSPORT);
}

for (const key of ["read", "write"]) {
  const seeded = registry.get(key);
  void seeded;
}

// An options object passed straight into a call. There is no declarator to
// qualify these keys against, so indexing them would produce owner-less
// symbols called `onError` and `onRetry`.
register("logging", {
  onError: () => Logger.warn("failed"),
  onRetry: () => null,
});

const Sequencer = class {
  next() {
    return null;
  }
};

const nextId = function* () {
  let n = 0;
  while (true) yield n++;
};

export { Transport as Wire, Frame } from "./protocol";
export * from "./errors";

export default function createRegistry() {
  return new Map();
}

module.exports = { Config, Client, buildClient };
