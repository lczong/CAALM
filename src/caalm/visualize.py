import json
import os
from datetime import datetime, timezone
from importlib.resources import files
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from . import __version__
from .io import round_nested_floats


def build_visualization_data(
    sequences: dict[str, str],
    all_ids: list[str],
    level0_map: dict[str, dict],
    level1_map: dict[str, dict],
    level2_map: dict[str, dict],
    level1_classes: list[str],
    output_name: str,
) -> dict:
    seq_list = []
    for seq_id in all_ids:
        seq = sequences.get(seq_id, "")
        l0 = level0_map.get(seq_id, {})
        l1 = level1_map.get(seq_id)
        l2 = level2_map.get(seq_id)

        entry = {
            "id": seq_id,
            "sequence": seq,
            "length": len(seq),
            "level0": {
                "pred_is_cazy": l0.get("pred_is_cazy", False),
                "prob_is_cazy": l0.get("prob_is_cazy", 0.0),
            },
            "level1": {
                "evaluated": l1 is not None,
                "predicted_classes": l1["predicted_classes"] if l1 else [],
                "class_probabilities": (
                    l1["class_probabilities"]
                    if l1
                    else {c: None for c in level1_classes}
                ),
            },
            "level2": {
                "evaluated": l2 is not None,
                "predicted_families": l2["predicted_families"] if l2 else [],
            },
        }
        seq_list.append(entry)

    # Compute statistics
    total = len(all_ids)
    cazy_count = sum(1 for s in seq_list if s["level0"]["pred_is_cazy"])

    level1_counts = {c: 0 for c in level1_classes}
    for s in seq_list:
        for c in s["level1"]["predicted_classes"]:
            level1_counts[c] = level1_counts.get(c, 0) + 1

    family_counts: dict[str, int] = {}
    for s in seq_list:
        for fam in s["level2"]["predicted_families"]:
            label = fam["family_label"]
            family_counts[label] = family_counts.get(label, 0) + 1

    data = {
        "metadata": {
            "output_name": output_name,
            "total_sequences": total,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": __version__,
        },
        "level1_classes": level1_classes,
        "sequences": seq_list,
        "statistics": {
            "level0": {
                "cazy": cazy_count,
                "non_cazy": total - cazy_count,
                "total": total,
            },
            "level1": level1_counts,
            "level2_families": family_counts,
        },
    }
    return round_nested_floats(data)


def render_report(data: dict) -> str:
    template_dir = str(files("caalm").joinpath("templates"))
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=False,
    )
    template = env.get_template("report.html")
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.render(data_json=data_json)


def write_report(
    sequences: dict[str, str],
    all_ids: list[str],
    level0_map: dict[str, dict],
    level1_map: dict[str, dict],
    level2_map: dict[str, dict],
    level1_classes: list[str],
    output_dir: str,
    output_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data = build_visualization_data(
        sequences=sequences,
        all_ids=all_ids,
        level0_map=level0_map,
        level1_map=level1_map,
        level2_map=level2_map,
        level1_classes=level1_classes,
        output_name=output_name,
    )
    html = render_report(data)
    report_path = os.path.join(output_dir, f"{output_name}_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML report to {report_path}")
