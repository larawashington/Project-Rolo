/**
 * Health inspection: the <FormRow> wrapper used by every Brain/Weather tab
 * render. Pure-presentational, but the className concat and the conditional
 * help/error slots have failure modes — these tests pin them so a refactor
 * that drops a slot or breaks the class composition fails loudly.
 */

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { FormRow } from "./FormRow";

describe("<FormRow>", () => {
  it("renders the label, the control, and no helper/error by default", () => {
    const { container } = render(
      <FormRow label="Server URL">
        <input data-testid="ctrl" />
      </FormRow>,
    );

    expect(screen.getByText("Server URL")).toBeInTheDocument();
    expect(screen.getByTestId("ctrl")).toBeInTheDocument();
    expect(container.querySelector(".cc-form-helper")).toBeNull();
    expect(container.querySelector(".cc-form-error")).toBeNull();
  });

  it("renders the help slot when `help` is truthy", () => {
    render(
      <FormRow label="API key" help="From your dashboard">
        <input />
      </FormRow>,
    );
    const helper = screen.getByText("From your dashboard");
    expect(helper).toBeInTheDocument();
    expect(helper.tagName).toBe("P");
    expect(helper).toHaveClass("cc-form-helper");
  });

  it("renders the error slot when `error` is truthy", () => {
    render(
      <FormRow label="API key" error="Required">
        <input />
      </FormRow>,
    );
    const err = screen.getByText("Required");
    expect(err).toBeInTheDocument();
    expect(err).toHaveClass("cc-form-error");
  });

  it("appends a custom className alongside the base `cc-form-row`", () => {
    const { container } = render(
      <FormRow label="x" className="cc-form-row-inline">
        <input />
      </FormRow>,
    );
    const row = container.firstChild as HTMLElement;
    expect(row.className).toBe("cc-form-row cc-form-row-inline");
  });

  it("falls back to the base class when no extra className is passed", () => {
    const { container } = render(
      <FormRow label="x">
        <input />
      </FormRow>,
    );
    const row = container.firstChild as HTMLElement;
    expect(row.className).toBe("cc-form-row");
  });

  it("wires the label's htmlFor to the underlying control id", () => {
    render(
      <FormRow label="Model" htmlFor="model-input">
        <input id="model-input" />
      </FormRow>,
    );
    const label = screen.getByText("Model") as HTMLLabelElement;
    expect(label.htmlFor).toBe("model-input");
  });

  it("renders both helper and error when both are provided", () => {
    render(
      <FormRow label="x" help="hint" error="bad">
        <input />
      </FormRow>,
    );
    expect(screen.getByText("hint")).toHaveClass("cc-form-helper");
    expect(screen.getByText("bad")).toHaveClass("cc-form-error");
  });
});
