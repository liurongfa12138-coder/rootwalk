# RootWalk Run Viewer

A read-only React viewer for a RootWalk `RunBundle`. It opens with the synthetic recorded bundle from `../../examples/bank-recorded-run`, can replace it with local `.json` and `.jsonl` files, and can load a second bundle for an in-browser comparison.

## Run

```bash
npm install
npm run dev
```

`predev` and `prebuild` copy the recorded example into `public/example-run/`; the source bundle is never modified.

## Verify

```bash
npm test
npm run build
```

The viewer runs entirely in the browser. Local files are parsed in memory and are neither uploaded nor changed. Select each bundle's JSON and JSONL files together; artifact payload files are optional for the M3 view. Comparison reports the first observable difference across environment manifest, context manifest, decision, observation, and termination without making a causal claim.
