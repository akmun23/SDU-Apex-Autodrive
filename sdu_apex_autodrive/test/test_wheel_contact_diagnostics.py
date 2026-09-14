import csv
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model_id.analyze_wheel_contact_diagnostics import (  # noqa: E402
    BASE_FIELDS,
    WHEEL_FIELDS,
    analyze,
)


def test_contact_diagnostic_analyzer_validates_complete_trace(tmp_path: Path):
    path = tmp_path / "wheel_contact_trace.csv"
    fields = list(BASE_FIELDS)
    for wheel in range(4):
        fields.extend(f"wheel{wheel}_{field}" for field in WHEEL_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(14):
            row = {field: "0" for field in fields}
            row.update({
                "fixed_time_s": f"{index * 0.01:g}",
                "fixed_step": str(index + 1),
                "render_frame": str(index + 1),
                "root_rotation_w": "1",
                "body_velocity_z_mps": "2",
                "controller_physics_step": str(index + 1),
            })
            for wheel in range(4):
                prefix = f"wheel{wheel}_"
                row[prefix + "grounded"] = "1"
                row[prefix + "contact_force_n"] = "10"
                row[prefix + "contact_normal_y"] = "1"
                row[prefix + "forward_slip"] = "0.01"
                row[prefix + "sideways_slip"] = "0.02"
                row[prefix + "rpm"] = "100"
            writer.writerow(row)

    report = analyze(path)

    assert report["status"] == "wheel_contact_diagnostics_analyzed"
    assert report["trace_schema"]["actual_columns"] == 110
    assert report["trace_schema"]["missing_columns"] == []
    assert report["timing"]["nonpositive_dt_rows"] == 0
    assert report["wheel_reports"]["wheel_0"]["grounded_fraction"] == 1.0
    assert report["wheel_reports"]["wheel_0"]["normal_load_proxy_n"][
        "mean"] == 10.0
    assert report["contact_load_transfer_diagnostics"]["samples"] == 14
    assert report["force_identification"]["separate_fx_fy_available"] is False
