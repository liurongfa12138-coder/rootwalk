import { useEffect, useRef } from "react";
import cytoscape, { type Core, type ElementDefinition, type StylesheetStyle } from "cytoscape";
import type { CausalPath, Topology } from "./types";

interface Props {
  topology: Topology;
  paths: CausalPath[];
  selectedEntity?: string;
  onSelectEntity: (entityId: string) => void;
}

const shortId = (id: string) => {
  const parts = id.split(":");
  return parts[parts.length - 1] || id;
};

export function InvestigationGraph({ topology, paths, selectedEntity, onSelectEntity }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const candidateIds = new Set(
      paths.flatMap((path) => [path.root_cause_entity_id, ...(path.symptom_entity_ids ?? [])]).filter(Boolean),
    );
    const elements: ElementDefinition[] = topology.nodes.map((node) => ({
      data: { id: node.canonical_id, label: shortId(node.canonical_id), kind: node.entity_type ?? "entity" },
      classes: candidateIds.has(node.canonical_id) ? "candidate" : "",
    }));
    elements.push(
      ...topology.edges.map((edge) => ({
        data: {
          id: `fact:${edge.edge_id}`,
          source: edge.source_id,
          target: edge.target_id,
          label: edge.relation_type,
          weight: edge.span_pairs ?? 1,
        },
        classes: "fact",
      })),
    );
    for (const path of paths) {
      for (const [index, step] of (path.steps ?? []).entries()) {
        if (!step.from_entity || !step.to_entity) continue;
        elements.push({
          data: {
            id: `path:${path.path_id ?? path.hypothesis_id ?? "candidate"}:${step.step_id ?? index}`,
            source: step.from_entity,
            target: step.to_entity,
            label: step.mechanism ?? "candidate",
          },
          classes: "candidate-path",
        });
      }
    }

    const style: StylesheetStyle[] = [
      {
        selector: "node",
        style: {
          "background-color": "#1b1a18",
          "border-color": "#1b1a18",
          "border-width": 1,
          color: "#1b1a18",
          "font-family": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
          "font-size": 8,
          height: 18,
          label: "data(label)",
          "text-background-color": "#eeeadf",
          "text-background-opacity": 0.88,
          "text-background-padding": "2px",
          "text-margin-y": -15,
          width: 18,
        },
      },
      {
        selector: "node.candidate",
        style: {
          "background-color": "#eeeadf",
          "border-color": "#c6311f",
          "border-width": 3,
          shape: "diamond",
          height: 24,
          width: 24,
        },
      },
      {
        selector: "node.selected",
        style: {
          "background-color": "#c6311f",
          "border-color": "#1b1a18",
          "border-width": 3,
          height: 28,
          width: 28,
        },
      },
      {
        selector: "edge.fact",
        style: {
          "curve-style": "bezier",
          "line-color": "#8a8478",
          "line-opacity": 0.62,
          "target-arrow-color": "#8a8478",
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.55,
          width: 1,
        },
      },
      {
        selector: "edge.candidate-path",
        style: {
          "curve-style": "unbundled-bezier",
          "line-color": "#c6311f",
          "line-style": "dashed",
          "line-dash-pattern": [7, 5],
          "target-arrow-color": "#c6311f",
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.75,
          width: 2,
        },
      },
    ];

    const graph = cytoscape({
      container: containerRef.current,
      elements,
      style,
      layout: { name: "cose", animate: false, fit: true, padding: 28, randomize: false },
      minZoom: 0.35,
      maxZoom: 2.5,
    });
    graph.on("tap", "node", (event) => onSelectEntity(event.target.id()));
    graphRef.current = graph;

    const observer = new ResizeObserver(() => {
      graph.resize();
      graph.fit(undefined, 28);
    });
    observer.observe(containerRef.current);
    return () => {
      observer.disconnect();
      graph.destroy();
      graphRef.current = null;
    };
  }, [topology, paths, onSelectEntity]);

  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    graph.nodes().removeClass("selected");
    if (selectedEntity) graph.getElementById(selectedEntity).addClass("selected");
  }, [selectedEntity]);

  return (
    <>
      <div
        className="graph-canvas"
        ref={containerRef}
        role="img"
        aria-describedby="graph-text-description"
        aria-label={`Investigation graph with ${topology.nodes.length} entities and ${topology.edges.length} observed topology edges.`}
      />
      <div className="sr-only" id="graph-text-description">
        <h3>Observed topology edges</h3>
        {topology.edges.length ? (
          <ul>
            {topology.edges.map((edge) => (
              <li key={edge.edge_id}>
                {shortId(edge.source_id)} {edge.relation_type} {shortId(edge.target_id)}.
              </li>
            ))}
          </ul>
        ) : <p>No observed topology edges recorded.</p>}
        <h3>Candidate causal steps</h3>
        {paths.some((path) => (path.steps ?? []).length > 0) ? (
          <ul>
            {paths.flatMap((path) => path.steps ?? []).map((step, index) => (
              <li key={step.step_id ?? `${step.from_entity}:${step.to_entity}:${index}`}>
                {shortId(step.from_entity)} to {shortId(step.to_entity)}: {step.mechanism ?? "mechanism not recorded"}; status {step.status ?? "not recorded"}.
              </li>
            ))}
          </ul>
        ) : <p>No candidate causal steps recorded.</p>}
      </div>
      <div className="graph-legend" aria-label="Graph legend">
        <span><i className="legend-fact" />Observed topology</span>
        <span><i className="legend-path" />Candidate path</span>
        <span><i className="legend-candidate" />Candidate entity</span>
      </div>
      <div className="entity-index" aria-label="Graph entities">
        {topology.nodes.map((node) => (
          <button
            aria-pressed={node.canonical_id === selectedEntity}
            className={node.canonical_id === selectedEntity ? "entity-button selected" : "entity-button"}
            key={node.canonical_id}
            onClick={() => onSelectEntity(node.canonical_id)}
            type="button"
          >
            {shortId(node.canonical_id)}
          </button>
        ))}
      </div>
    </>
  );
}
