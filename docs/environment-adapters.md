# Environment adapter boundary

RootWalk keeps environment integration behind four structural interfaces:

- `TaskSource` exposes the visible `Task` and frozen dataset identity.
- `TopologyProvider` exposes the source-backed `TopologySnapshot`.
- `ToolProvider` exposes the bounded, read-only query surface.
- `ResultExporter` writes the selected result format.

The environment boundary is the composition of these protocols plus a
`CapabilityManifest`; there is no wrapper object, registry, or plugin
lifecycle. Evaluators, private labels, and scoring code are deliberately
outside this boundary.

`BankEnvironment` implements the runtime-facing Task, Topology, and Tool
protocols. Its capability manifest declares Metrics, Logs, Trace, and
Topology support, together with the frozen-slice coverage limits. The existing
`BundleExporter` implements `ResultExporter` without an adapter-specific copy.
