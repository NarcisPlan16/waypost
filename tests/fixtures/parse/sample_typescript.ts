/**
 * Fixture: a plausible TypeScript module.
 *
 * Half the forms here (enum, type alias, namespace, exported const) are the
 * ones the bundled TypeScript tags query does not capture, which is exactly
 * why they are in the fixture.
 */

import { Logger } from "./logger";
import * as http from "node:http";

export const DEFAULT_TIMEOUT = 30;
const INTERNAL_RETRIES = 3;

export type RequestBody = string | Uint8Array | null;

export type Handler = (req: Request, res: Response) => void;

export enum Method {
  Get = "GET",
  Post = "POST",
}

export interface Shape {
  area(): number;
  perimeter(): number;
}

export interface Transport {
  send(body: RequestBody): Promise<number>;
}

/** Runtime configuration. */
export class Config {
  timeout: number;

  constructor(timeout: number) {
    this.timeout = timeout;
  }

  /** Serialise to a plain object. */
  toObject(): Record<string, number> {
    return { timeout: this.timeout };
  }

  static fromEnv(): Config {
    return new Config(DEFAULT_TIMEOUT);
  }

  get isDefault(): boolean {
    return this.timeout === DEFAULT_TIMEOUT;
  }
}

/** HTTP client with retry and pooling. */
export class Client implements Transport {
  private readonly log: Logger;

  constructor(log: Logger) {
    this.log = log;
  }

  /** Issue a request, retrying on transient failures. */
  send(body: RequestBody): Promise<number> {
    return this.retry(body);
  }

  private retry(body: RequestBody): Promise<number> {
    return http.request(body);
  }

  close(): void {
    this.log.info("closed");
  }
}

export abstract class Backend {
  abstract connect(url: string): void;
}

export function buildClient(log: Logger): Client {
  return new Client(log);
}

export function describe(shape: Shape): string {
  return `${shape.area()}`;
}

function normalise(url: string): string {
  return url.trim();
}

export const parseUrl = (raw: string): string => normalise(raw);

export const VERSION = "0.1.0";

export interface Options {
  retries: number;
}

export function withDefaults(opts: Partial<Options>): Options {
  return { retries: INTERNAL_RETRIES, ...opts };
}

// The everyday TS idiom for a dispatch table. `as const` sits between the
// object and its declarator and must not hide the keys.
export const routes = {
  home: () => "/",
  about: () => "/about",
} as const;

export namespace Internals {
  export function checksum(data: string): number {
    return data.length;
  }

  export const SEED = 7;

  export namespace Deep {
    export const NESTED_SEED = 11;
  }
}

export { Codec as Wire, Frame } from "./protocol";
export * from "./errors";
export * as Legacy from "./legacy";

if (typeof process !== "undefined") {
  const NODE_ONLY = true;
  void NODE_ONLY;
}

export default class Session {
  open(): void {}
}
