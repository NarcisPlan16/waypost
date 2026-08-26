/**
 * Fixture: a plausible TSX module.
 *
 * The point of a separate TSX fixture is that `tsx` is its own grammar, not
 * an alias for `typescript`: JSX in the body must not stop the surrounding
 * declarations from being found.
 */

import * as React from "react";
import { Logger } from "./logger";

export const DEFAULT_LABEL = "Submit";
const INTERNAL_CLASS = "wp-btn";

export type Variant = "primary" | "ghost";

export type ClickHandler = (event: React.MouseEvent) => void;

export enum Size {
  Small = "sm",
  Large = "lg",
}

export interface ButtonProps {
  label: string;
  onClick(event: React.MouseEvent): void;
}

export interface PanelProps {
  title: string;
  render(): React.ReactNode;
}

/** A styled button. */
export function Button(props: ButtonProps): React.ReactElement {
  return <button className={INTERNAL_CLASS}>{props.label}</button>;
}

export function Panel(props: PanelProps): React.ReactElement {
  return (
    <section>
      <h2>{props.title}</h2>
      {props.render()}
    </section>
  );
}

export const Badge = (props: { text: string }): React.ReactElement => (
  <span>{props.text}</span>
);

/** Stateful widget. */
export class Toolbar extends React.Component<PanelProps> {
  private readonly log: Logger;

  constructor(props: PanelProps) {
    super(props);
    this.log = new Logger();
  }

  /** Render the toolbar and its children. */
  render(): React.ReactElement {
    return <div>{this.props.title}</div>;
  }

  private track(name: string): void {
    this.log.info(name);
  }

  get label(): string {
    return DEFAULT_LABEL;
  }
}

export abstract class Surface {
  abstract paint(ctx: CanvasRenderingContext2D): void;
}

export function useLabel(variant: Variant): string {
  return variant === "primary" ? DEFAULT_LABEL : "";
}

function classNames(...parts: string[]): string {
  return parts.join(" ");
}

export interface Theme {
  color(variant: Variant): string;
  spacing(size: Size): number;
}

export const THEME_KEY = "wp-theme";

export function useTheme(): Theme {
  return { color: () => "", spacing: () => 0 };
}

export const VERSION = "0.1.0";

export function withDefaults(props: Partial<ButtonProps>): ButtonProps {
  return { label: DEFAULT_LABEL, onClick: () => undefined, ...props };
}

export namespace Internals {
  export function checksum(data: string): number {
    return classNames(data).length;
  }

  export const SEED = 7;
}
