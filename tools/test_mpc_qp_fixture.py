import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools import mpc_qp_fixture as fixture


def make_valid_payload():
    data = bytearray(fixture.MAGIC)
    data += struct.pack("<6Ii", fixture.SCHEMA_VERSION, 0x01020304,
                        fixture.NX, fixture.NU, fixture.HORIZON, 1, 0)

    def floats(count, value=0.0):
        data.extend(struct.pack(f"<{count}f", *([value] * count)))

    def ints(count, value=0):
        data.extend(struct.pack(f"<{count}i", *([value] * count)))

    nx, nu, horizon = fixture.NX, fixture.NU, fixture.HORIZON
    for _ in range(horizon):
        floats(nx * nx)
        floats(nx * nu)
        floats(nx)
        floats(nx)
        floats(nx)
        floats(nx * nu)
        floats(nu)
        floats(nu)
        floats(nx, -1.0)
        floats(nx, 1.0)
        floats(nu, -1.0)
        floats(nu, 1.0)
    floats(nx)
    floats(nx)
    floats(nx, -1.0)
    floats(nx, 1.0)
    floats(nx)

    floats(2, 1.0)  # rho, rho_u
    floats(nx)  # default per-state penalties
    floats(nu)  # default per-input penalties
    floats(2, 1.0)  # rho_max, solver tolerance
    ints(5, 1)  # max iter, adaptive, cold, update period, shared rho
    floats(1, 1.5)  # over-relaxation
    data.extend(struct.pack("<d", 0.0))
    ints(1)  # scaling enabled
    floats(nx, 1.0)
    floats(nu, 1.0)

    floats((horizon + 1) * nx)
    floats(horizon * nu)
    floats((horizon + 1) * nx)
    floats(horizon * nu)
    floats(2)
    floats((horizon + 1) * nx)
    floats(horizon * nu)
    ints(2)
    floats(nx)
    floats(nu)

    floats((horizon + 1) * nx)
    floats(horizon * nu)
    floats((horizon + 1) * nx)
    floats(horizon * nu)
    floats(2)
    floats((horizon + 1) * nx)
    floats(horizon * nu)
    ints(2)
    floats(nx)
    floats(nu)

    floats(horizon * nu)

    floats((horizon + 1) * nx)
    floats(horizon * nu)
    ints(2)
    floats(6)
    ints(8, -1)
    ints(1)
    floats(4)
    ints(1)
    floats(5)
    ints(1)
    floats(1)
    ints(2)
    data.extend(struct.pack("<7Q", *([0] * 7)))
    return bytes(data)


class FrozenQpFixtureTest(unittest.TestCase):
    def test_roundtrip_schema_and_summary(self):
        summary = fixture.parse_payload(make_valid_payload())
        self.assertEqual(summary["nx"], 22)
        self.assertEqual(summary["nu"], 2)
        self.assertEqual(summary["qp_horizon"], 29)
        self.assertEqual(summary["capture_passes"], 1)
        self.assertGreater(summary["float_count"], 20_000)

    def test_rejects_bad_magic_and_dimensions(self):
        payload = bytearray(make_valid_payload())
        payload[0] ^= 0xFF
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(payload)
        payload = bytearray(make_valid_payload())
        struct.pack_into("<I", payload, 16, 21)
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(payload)

    def test_rejects_truncation_trailing_bytes_and_nonfinite(self):
        payload = make_valid_payload()
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(payload[:-1])
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(payload + b"x")
        payload = bytearray(payload)
        # First element of stage A begins immediately after the 36-byte header.
        struct.pack_into("<f", payload, 36, float("nan"))
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(payload)

    def test_rejects_corrupt_bounds(self):
        payload = bytearray(make_valid_payload())
        # A stage's x_lb follows A, B, d, Q, q, and N.
        stage_x_lb_offset = 36 + 4 * (
            fixture.NX * fixture.NX + 2 * fixture.NX * fixture.NU +
            3 * fixture.NX + 2 * fixture.NU)
        struct.pack_into("<f", payload, stage_x_lb_offset, 2.0)
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(payload)

    def test_sidecar_hash_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "qp.fqp"
            sidecar = Path(directory) / "qp.json"
            data = make_valid_payload()
            payload.write_bytes(data)
            metadata = {
                "payload_sha256": hashlib.sha256(data).hexdigest(),
                "canonical_qp_fixture_sha256": hashlib.sha256(data).hexdigest(),
                "dimensions": {"nx": 22, "nu": 2, "qp_horizon": 29},
                "source_abi": {
                    "model": {"files": ["model.c"], "sha256": "a" * 64},
                    "solver_and_scaling": {
                        "files": ["solver.c"], "sha256": "b" * 64},
                },
            }
            sidecar.write_text(json.dumps(metadata))
            fixture.validate_sidecar(payload, sidecar)
            metadata["source_abi"]["model"]["sha256"] = "invalid"
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaises(fixture.FixtureError):
                fixture.validate_sidecar(payload, sidecar)
            payload.write_bytes(data + b"corruption")
            with self.assertRaises(fixture.FixtureError):
                metadata["source_abi"]["model"]["sha256"] = "a" * 64
                sidecar.write_text(json.dumps(metadata))
                fixture.validate_sidecar(payload, sidecar)


if __name__ == "__main__":
    unittest.main()
