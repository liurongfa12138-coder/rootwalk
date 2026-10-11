---
name: RootWalk Run Viewer
description: A rigorous evidence folio for inspecting recorded investigations.
colors:
  vermilion: "#c6311f"
  unbleached-bone: "#eeeadf"
  paper: "#f7f3e8"
  iron: "#1b1a18"
  muted-iron: "#676158"
  rule: "#aaa397"
  soft-rule: "#d2ccc0"
typography:
  display:
    fontFamily: "Barlow Condensed Local, sans-serif"
    fontSize: "clamp(2.2rem, 4vw, 3.8rem)"
    fontWeight: 700
    lineHeight: 0.85
    letterSpacing: "-0.04em"
  body:
    fontFamily: "Arial, sans-serif"
    fontSize: "0.95rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "0.69rem"
    fontWeight: 700
    lineHeight: 1.45
rounded:
  none: "0"
spacing:
  compact: "10px"
  section: "22px"
components:
  bundle-button:
    backgroundColor: "{colors.iron}"
    textColor: "{colors.unbleached-bone}"
    rounded: "{rounded.none}"
    padding: "9px 12px 8px"
  selected-record:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.vermilion}"
    rounded: "{rounded.none}"
    padding: "9px 5px"
---

# Design System: RootWalk Run Viewer

## Overview

**Creative North Star: "The Investigation Folio"**

RootWalk reads like a precise, annotated engineering record: AI calls, returned decisions, resource use, and uncertainty lead into the supporting evidence. The interface is compact and flat, using alignment and rulings instead of card chrome.

**Key Characteristics:**

- A round navigator beside call detail on wide screens, above it on narrow screens.
- Bone paper, iron text, and rare vermilion annotations.
- Explicit status labels, with geometric marks in the evidence view.
- Six-layer run comparison is a ruled evidence band, never a metric dashboard.

## Colors

Vermilion is reserved for selection, uncertainty, and candidate annotations. Bone and paper separate fields without shadows; iron rules establish structure.

**The Annotation Rule.** Vermilion explains a state or relationship; it never becomes decorative fill.

## Typography

Barlow Condensed carries run identity and section authority. Arial keeps long evidence readable. Monospace is limited to revisions, IDs, selectors, labels, and measurements.

**The Evidence Voice Rule.** Prose stays sans-serif; machine-originated identity and measurement stay monospace.

## Layout

Compact run identity and termination precede the AI call panel. Long identifiers and termination text retain their complete values in disclosures. Wide screens place the round navigator in a 28% column (220px minimum) beside the selected round. Below 820px, rounds become a horizontally scrollable strip above a single detail column. Call sections use one-pixel dividers and 22px horizontal padding, reduced to 16px on narrow screens.

The collapsed “证据与调查图” disclosure follows call inspection. When opened, it retains two evidence columns around a narrower sticky revision rail. Below 820px the rail moves first and evidence becomes one reading column; below 520px padding reduces to 14–16px and paired evidence relationships stack.

Optional run comparison sits between the run summary and call panel. It exposes all six protocol layers in order: environment manifest, context manifest, decision, observation, evidence, and termination, followed by the first differing values.

## Elevation & Depth

The system uses no shadows. Depth comes only from paper tones, position, and one-pixel rules.

## Shapes

Controls and containers use zero radius. Squares indicate ordinary state, double-stroked squares indicate partial state, diamonds indicate candidate evidence, and dashed lines indicate candidate causal paths.

## Components

### Buttons

Buttons are rectangular, compact, and text-led. Focus uses a two-pixel vermilion outline with a three-pixel offset. Selected evidence rows retain their content and gain vermilion text plus an inset rule.

### AI Round Navigator

Each keyboard-operable button shows the round number, a two-line decision summary, recorded Token usage, duration, status, and tool count. The selected round uses iron fill with paper text and `aria-pressed`. Selection replaces the adjacent round detail; the run totals stay above the navigator.

### Call Detail

Ruled definition rows separate duration, input/output Tokens, cache hits, reasoning Tokens, and provider request count. The context meter shows estimated compiled-context Tokens against the configured budget, with a numeric percentage and an accessible meter value; its eight-pixel iron fill is capped at the track width. The estimate is labeled separately from provider-reported input usage. Missing measurements remain “未记录”, and an unknown budget does not produce a meter.

Decision summaries lead into planned operations, model input/output, individual provider requests, tool execution, and Runtime records. Provider repair or format-fallback requests retain their own usage, duration, reason, and input/output; failed request details open by default. Only reasoning text actually returned by the model is displayed.

### Artifact Disclosures

Native disclosures load local artifact content when opened, with loading, failure, and retry states. Messages retain role labels; returned analysis text is separately expandable. Raw values render as escaped text in bounded scrolling regions, and complete artifacts remain available through JSON download. Missing content and legacy metadata stay explicitly unknown.

### Revision Rail

Inside the evidence disclosure, each revision is a keyboard-operable button aligned to one vertical rule. Mark shape and an explicit event label communicate type without relying on color.

### Evidence Detail

Definition rows expose subject, interval, value, and uncertainty. Source selectors and coverage render as escaped preformatted text; support, opposition, and gap remain separately labeled.

## Do's and Don'ts

### Do:

- **Do** keep termination and limits visible before investigation detail.
- **Do** distinguish recorded Token usage from context estimates, and unknown metadata from zero or disabled states.
- **Do** distinguish observed topology from candidate causal paths by line style and label.
- **Do** preserve one-pixel rulings and zero-radius controls across new states.

### Don't:

- **Don't** convert the page into a card grid or add decorative metrics.
- **Don't** use color as the only status cue.
- **Don't** render bundle content as HTML or imply that grounding proves root-cause correctness.
