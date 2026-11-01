"""CPU figures for frozen attribution-logit policies and their pruning masks.

Colors show sigmoid thresholds for eligible retained edges, not the marginal
frequency of an edge after spatial cycle rejection. No graph is sampled here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments_attrprune.visualize_pruning_matrices import AGENT_DISPLAY_NAMES

BLOCKS = ((0, "spatial"), (1, "spatial"), (1, "temporal"))
SEMANTICS = "Sigmoid Bernoulli thresholds after masking/cycle eligibility, not realized spatial inclusion frequencies"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_policies(root: Path, scopes: Sequence[str], percentages: Sequence[int]) -> tuple[dict, dict]:
    """Read frozen policies/configs without touching evaluation checkpoints."""
    if not scopes or len(set(scopes)) != len(scopes) or set(scopes) - {"global", "uniform"}:
        raise ValueError("Choose unique global/uniform scopes")
    if not percentages or len(set(percentages)) != len(percentages) or set(percentages) - {50, 60}:
        raise ValueError("Choose unique 50/60 percentages")
    policies, hashes = {}, {}
    for scope in scopes:
        for rate in percentages:
            folder = Path(root) / scope / f"prune_{rate}pct"
            path, config_path = folder / "sampling_policy.json", folder / "evaluation_config.json"
            policy, config = [json.loads(p.read_text()) for p in (path, config_path)]
            if _digest(policy) != config.get("sampling_policy_sha256"):
                raise ValueError(f"Frozen policy hash changed: {path}")
            if (policy.get("policy_type") != "attribution_logits_agentprune_sampling"
                    or policy.get("pruning_scope") != scope or policy.get("pruning_rate") != rate / 100
                    or policy.get("round_policy") != "per_round" or policy.get("num_rounds") != 2
                    or policy.get("matrix_orientation") != "[round][source][target]"
                    or policy.get("probability_mapping", {}).get("name") != "importance_as_logit"):
                raise ValueError(f"Unexpected sampled policy definition: {path}")
            agents = policy["agents"]
            if not agents or len(set(agents)) != len(agents):
                raise ValueError("Policy requires unique canonical agent identities")
            n = len(agents)
            for edge in ("spatial", "temporal"):
                params = policy["parameters"][edge]
                arrays = {key: np.asarray(params[key], dtype=float) for key in
                          ("importance", "logits", "probabilities", "candidate_mask", "effective_probabilities")}
                arrays["mask"] = np.asarray(policy["masks"][edge], dtype=float)
                if any(a.shape != (2, n, n) or not np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError("Invalid sampled matrix shape or nonfinite value")
                importance, logits, probabilities = (arrays[k] for k in ("importance", "logits", "probabilities"))
                candidate, mask = (arrays[k] for k in ("candidate_mask", "mask"))
                expected_candidates = np.ones((2, n, n), dtype=int)
                if edge == "spatial":
                    expected_candidates[:, np.arange(n), np.arange(n)] = 0
                else:
                    expected_candidates[0] = 0
                if (not np.array_equal(candidate, expected_candidates)
                        or not np.isin(mask, (0, 1)).all() or np.any(mask > candidate)):
                    raise ValueError("Masks violate spatial/temporal candidate topology")
                if ((importance < 0).any() or (importance > 1).any()
                        or not np.array_equal(logits, importance.astype(np.float32).astype(float))
                        or not np.allclose(probabilities, 1 / (1 + np.exp(-logits)), atol=1e-7, rtol=0)
                        or not np.array_equal(arrays["effective_probabilities"], probabilities * mask)):
                    raise ValueError("Probabilities must equal sigmoid(importance) with the frozen mask")
            policies[scope, rate] = policy
            hashes.update({str(p.resolve()): _sha(p) for p in (path, config_path)})
    base = next(iter(policies.values()))
    for policy in policies.values():
        if policy["agents"] != base["agents"] or policy.get("source") != base.get("source"):
            raise ValueError("Comparison policies must share their calibration source and identities")
        for edge in ("spatial", "temporal"):
            for key in ("importance", "logits", "candidate_mask"):
                if policy["parameters"][edge][key] != base["parameters"][edge][key]:
                    raise ValueError("Comparison policies must use the same unpruned matrices")
    return policies, hashes


def render(evaluation_root: Path, output_dir: Path, *, scopes: Sequence[str] = ("global", "uniform"),
           percentages: Sequence[int] = (50, 60), dpi: int = 180) -> dict:
    policies, hashes = load_policies(evaluation_root, scopes, percentages)
    output_dir = Path(output_dir).resolve()
    if any(output_dir == Path(p).parent for p in hashes):
        raise ValueError("Use a separate visualization directory, not a checkpoint directory")
    if dpi < 1:
        raise ValueError("DPI must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle

    base = next(iter(policies.values()))
    agents = base["agents"]
    labels = [AGENT_DISPLAY_NAMES.get(a, a) for a in agents]
    n = len(agents)
    norm = Normalize(.5, .75)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f2f5")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    def panel(axis, policy, r, edge, *, unpruned=False, other=None):
        params = policy["parameters"][edge]
        candidate = np.asarray(params["candidate_mask"][r], dtype=bool)
        mask = candidate if unpruned else np.asarray(policy["masks"][edge][r], dtype=bool)
        values = np.asarray(params["probabilities"][r])
        exclusive = np.zeros_like(mask) if other is None else mask & ~np.asarray(other["masks"][edge][r], dtype=bool)
        axis.imshow(np.ma.masked_where(~mask, values), cmap=cmap, norm=norm, interpolation="nearest")
        heading = "Before pruning" if unpruned else f"{policy['pruning_scope'].title()} · {policy['pruning_rate']*100:g}% pruned"
        axis.set_title(f"{heading}\nRound {r} · {edge} · kept {mask.sum()}/{candidate.sum()}", fontsize=12)
        axis.set_xticks(range(n), labels=labels, rotation=30, ha="right", fontsize=11)
        axis.set_yticks(range(n), labels=labels, fontsize=11)
        axis.set_xlabel(f"target · round {r}", fontsize=11)
        axis.set_ylabel(f"source · round {r if edge == 'spatial' else r-1}", fontsize=11)
        for i in range(n):
            for j in range(n):
                if not candidate[i, j]:
                    axis.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor="#e2e7ef",
                                             edgecolor="#c2cad5", hatch="//", linewidth=.3))
                    label, color = "0", "#647084"
                elif not mask[i, j]:
                    label, color = "0", "#8a929e"
                else:
                    label = f"{values[i,j]:.3f}"
                    color = "#132338" if norm(values[i,j]) > .65 else "white"
                if exclusive[i, j]:
                    axis.add_patch(Rectangle((j-.465, i-.465), .93, .93,
                                             facecolor="none", edgecolor="#ed8b16", linewidth=2.5))
                axis.text(j, i, label, ha="center", va="center", fontsize=11, color=color)

    def save(stem, title, cells, *, comparison=False):
        rows, columns = len(cells), len(cells[0])
        fig, axes = plt.subplots(rows, columns, figsize=(5.2*columns+1, 4.9*rows+1.8), squeeze=False)
        try:
            fig.subplots_adjust(left=.09, right=.86, top=.84 if rows == 1 else .90,
                                bottom=.24 if rows == 1 else .12, wspace=.42, hspace=.55)
            for row, items in enumerate(cells):
                for col, item in enumerate(items):
                    policy, r, edge, unpruned, other = item
                    panel(axes[row,col], policy, r, edge, unpruned=unpruned, other=other)
            fig.suptitle(title + "\np = sigmoid(attribution importance)", fontsize=17, y=.98)
            color_axis = fig.add_axes([.90, .32, .014, .38])
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis).set_label(
                "Bernoulli probability for eligible retained edges")
            legend = [Patch(facecolor="#f0f2f5", label="Pruned: p = 0"),
                      Patch(facecolor="#e2e7ef", hatch="//", label="Forbidden spatial self-link")]
            if comparison:
                legend.insert(0, Patch(facecolor="none", edgecolor="#ed8b16", linewidth=2.5,
                                       label="Retained only in this policy"))
            fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5,.055),
                       ncol=len(legend), frameon=False, fontsize=10)
            fig.text(.5,.035,"Rows: source; columns: target. Round 0 temporal is absent; reviewer inputs remain dense.",
                     ha="center", fontsize=9)
            fig.text(.5,.015,"Spatial cycle checks precede Bernoulli draws; displayed values are not realized inclusion frequencies.",
                     ha="center", fontsize=9)
            figures[stem] = {}
            for extension in ("png", "svg"):
                path = output_dir / f"{stem}.{extension}"
                temporary = path.with_name(f"{stem}.tmp.{extension}")
                fig.savefig(temporary, dpi=dpi, facecolor="white")
                temporary.replace(path)
                figures[stem][extension] = str(path)
        finally:
            plt.close(fig)

    with plt.rc_context({"font.family":"DejaVu Sans", "svg.fonttype":"none"}):
        save("sigmoid_before_pruning", "MMLU · before pruning",
             [[(base, r, edge, True, None) for r,edge in BLOCKS]])
        for scope in scopes:
            save(f"sigmoid_{scope}_{'_'.join(map(str,percentages))}", f"MMLU · {scope} pruning + sigmoid sampling",
                 [[(policies[scope,rate], r, edge, False, None) for r,edge in BLOCKS] for rate in percentages])
        if len(scopes) == 2:
            for rate in percentages:
                save(f"sigmoid_{rate}pct_global_vs_uniform", f"MMLU · {rate}% global vs uniform pruning",
                     [[(policies[scope,rate],r,edge,False,policies[scopes[1-i],rate])
                       for i,scope in enumerate(scopes)] for r,edge in BLOCKS], comparison=True)

    matrices, records = {}, []
    for scope, rate, policy in [("unpruned",0,base)] + [(s,p,v) for (s,p),v in policies.items()]:
        key = f"{scope}_{rate}pct"
        matrices[key] = {}
        for edge in ("spatial", "temporal"):
            params = policy["parameters"][edge]
            candidate = np.asarray(params["candidate_mask"], dtype=bool)
            retained = candidate if scope == "unpruned" else np.asarray(policy["masks"][edge], dtype=bool)
            probabilities = np.asarray(params["probabilities"])
            effective = probabilities * retained
            matrices[key][edge] = {"importance":params["importance"], "logits":params["logits"],
                "sigmoid_probability":probabilities.tolist(), "candidate_mask":candidate.astype(int).tolist(),
                "retained_mask":retained.astype(int).tolist(), "effective_probability":effective.tolist()}
            for r in range(2):
                for i,source in enumerate(agents):
                    for j,target in enumerate(agents):
                        records.append({"scope":scope, "pruning_percent":rate,"round":r,"edge_type":edge,
                            "source":source,"target":target,"importance":params["importance"][r][i][j],
                            "logit":params["logits"][r][i][j],"candidate":int(candidate[r,i,j]),
                            "retained":int(retained[r,i,j]),
                            "status":"retained" if retained[r,i,j] else ("pruned" if candidate[r,i,j] else "unavailable"),
                            "probability_if_eligible":float(probabilities[r,i,j]) if candidate[r,i,j] else "",
                            "effective_probability":float(effective[r,i,j])})
    csv_path, json_path = output_dir/"sigmoid_edges.csv", output_dir/"sigmoid_matrices.json"
    with csv_path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    json_path.write_text(json.dumps({"agents":agents,"display_labels":labels,
        "matrix_orientation":"[round][source][target]","semantics":SEMANTICS,"matrices":matrices},indent=2)+"\n")
    for name, expected in hashes.items():
        if _sha(Path(name)) != expected:
            raise ValueError(f"Frozen input changed during rendering: {name}")
    manifest = {"complete":True,"source_sha256":hashes,"renderer_sha256":_sha(Path(__file__)),
        "semantics":SEMANTICS,"color_limits":[.5,.75],"agents":agents,"display_labels":labels,
        "figures":figures,"csv":str(csv_path),"json":str(json_path)}
    (output_dir/"visualization_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True,
                        help="Directory containing global/prune_50pct and uniform/prune_50pct, etc.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scopes", nargs="+", choices=("global","uniform"), default=["global","uniform"])
    parser.add_argument("--percentages", nargs="+", type=int, choices=(50,60), default=[50,60])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    result = render(args.evaluation_root,args.output_dir,scopes=args.scopes,percentages=args.percentages,dpi=args.dpi)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
