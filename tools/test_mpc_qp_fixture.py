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
        floats(nx * nx)  # Full offdiagonal Q (schema v5)
        floats(nx * nu)
        floats(nu)
        floats(nu)
        floats(1)  # R cross term (schema v5)
        floats(nx, -1.0)
        floats(nx, 1.0)
        floats(nu, -1.0)
        floats(nu, 1.0)
    floats(nx)
    floats(nx)
    floats(nx, -1.0)
    floats(nx, 1.0)
    floats(nx)

    for _ in range(horizon + 1):
        ints(1)  # no coupled footprint planes in this generic fixture
        floats(8 * 3)

    floats(2, 1.0)  # rho, rho_u
    floats(nx)  # default per-state penalties
    floats(nu)  # default per-input penalties
    floats(2, 1.0)  # rho_max, solver tolerance
    ints(6, 1)  # max iter, adaptive, cold, same epoch, update period, shared rho
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
        self.assertEqual(summary["nx"], 23)
        self.assertEqual(summary["nu"], 2)
        self.assertEqual(summary["qp_horizon"], 30)
        self.assertEqual(summary["capture_passes"], 1)
        self.assertEqual(summary["reuse_same_epoch"], 1)
        self.assertGreater(summary["float_count"], 20_000)

    def test_independent_oracle_reconstructs_full_coupled_hessian(self):
        # The old fixture schema dropped the cross-state and cross-control
        # Hessian entirely, so prior independent convergence experiments
        # solved a DIFFERENT QP while reporting it as exact reproduction.
        from tools.mpc_qp_oracle import decode_problem, build_osqp_matrices
        data = bytearray(make_valid_payload())
        nx,nu=fixture.NX,fixture.NU
        start=fixture.HEADER_SIZE
        qoff=start+4*(nx*nx+nx*nu+3*nx)
        # Each Qoff entry carries the COMPLETE symmetric matrix, not an
        # upper-triangle compression. 0.75 is intentionally signed.
        struct.pack_into('<f',data,qoff+4*(2*nx+4),-0.75)
        struct.pack_into('<f',data,qoff+4*(4*nx+2),-0.75)
        rcross=qoff+4*(nx*nx+nx*nu+2*nu)
        struct.pack_into('<f',data,rcross,.125)
        prob=decode_problem(bytes(data))
        P,*_=build_osqp_matrices(prob)
        xsize=(fixture.HORIZON+1)*nx
        self.assertAlmostEqual(P[2,4],-.75,places=6)
        self.assertAlmostEqual(P[xsize,xsize+1],.125,places=6)
        self.assertEqual(prob["steps"][0][12].shape,(23,23))
        self.assertAlmostEqual(prob["steps"][0][13],.125,places=6)

    def test_rejects_bad_magic_and_dimensions(self):
        payload = bytearray(make_valid_payload())
        payload[0] ^= 0xFF
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(payload)
        payload = bytearray(make_valid_payload())
        # Retired 22-state captures lack the propagated progress coordinate.
        struct.pack_into("<I", payload, 16, 22)
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
            2 * fixture.NX * fixture.NX + 2 * fixture.NX * fixture.NU +
            3 * fixture.NX + 2 * fixture.NU + 1)
        struct.pack_into("<f", payload, stage_x_lb_offset, 2.0)
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(payload)

    def test_polygon_metadata_is_checked_and_old_schema_rejected(self):
        nx, nu = fixture.NX, fixture.NU
        start = fixture.HEADER_SIZE + 4*(fixture.HORIZON*(2*nx*nx+2*nx*nu+5*nx+4*nu+1)+5*nx)
        data = bytearray(make_valid_payload())
        struct.pack_into('<i', data, start, 9)
        with self.assertRaisesRegex(fixture.FixtureError, 'plane count'):
            fixture.parse_payload(data)
        struct.pack_into('<i', data, start, 1)
        with self.assertRaisesRegex(fixture.FixtureError, 'normal'):
            fixture.parse_payload(data)
        struct.pack_into('<3f', data, start+4, 1., .43, .2)
        self.assertEqual(fixture.parse_payload(data)['state_polygon_plane_count'], 1)
        data[:8] = b'SDUFQP3\n'
        with self.assertRaises(fixture.FixtureError):
            fixture.parse_payload(data)

    def test_sidecar_hash_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "qp.fqp"
            sidecar = Path(directory) / "qp.json"
            data = make_valid_payload()
            payload.write_bytes(data)
            metadata = {
                "payload_sha256": hashlib.sha256(data).hexdigest(),
                "canonical_qp_fixture_sha256": hashlib.sha256(data).hexdigest(),
                "dimensions": {"nx": 23, "nu": 2, "qp_horizon": 30},
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



class TerminalConstraintExtensionTests(unittest.TestCase):
    def payload(self):
        normal = [0.0] * fixture.NX
        normal[2], normal[4] = .25, -.4
        return (make_valid_payload() + struct.pack('<Iif', 0x314d5254, 1, 7.) +
                struct.pack(f'<{fixture.NX + 1}f', *normal, .2))

    def test_terminal_planes_roundtrip_and_reach_independent_oracle(self):
        from tools.mpc_qp_oracle import decode_problem, build_osqp_matrices
        parsed = fixture.parse_payload(self.payload())
        self.assertEqual(len(parsed['terminal_constraints']), 1)
        self.assertAlmostEqual(parsed['terminal_constraints'][0]['normal'][2], .25)
        problem = decode_problem(self.payload())
        _, _, matrix, lower, upper = build_osqp_matrices(problem)
        row = matrix.getrow(matrix.shape[0] - 1)
        self.assertEqual(row.nnz, 2)
        self.assertAlmostEqual(row[0, fixture.HORIZON * fixture.NX + 2], .25)
        self.assertAlmostEqual(row[0, fixture.HORIZON * fixture.NX + 4], -.4)
        self.assertAlmostEqual(upper[-1], .2)
        self.assertEqual(lower[-1], float('-inf'))

    def test_malformed_extension_fails_closed(self):
        payload = self.payload()
        for bad in [payload[:-1], payload + b'x',
                    make_valid_payload() + struct.pack('<Iif', 0x314d5254, 11, 7.),
                    make_valid_payload() + struct.pack('<Iif', 0x314d5254, 1, -1.)]:
            with self.subTest(size=len(bad)):
                with self.assertRaises(fixture.FixtureError):
                    fixture.parse_payload(bad)


if __name__ == "__main__":
    unittest.main()
