"""Build the offline shim against the production MPC core C sources."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CORE_SOURCES = (
    "mpc_reference.c",
    "vehicle_model.c",
    "mpc_linearization.c",
    "mpc_rti.c",
    "riccati_solver.c",
)


def build(repo: Path, output: Path, compiler: str = "cc",
          cxx_compiler: str = "c++") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    object_files = []
    for name in CORE_SOURCES:
        obj = output.parent / f"{Path(name).stem}.o"
        command = [
            compiler, "-std=c99", "-O3", "-fPIC",
            "-D_POSIX_C_SOURCE=200809L", "-DMPC_ENABLE_RICCATI_PROFILE",
            "-I", str(repo / "f1tenth_mpc/include"),
            "-c", str(repo / "f1tenth_mpc/src" / name),
            "-o", str(obj),
        ]
        subprocess.run(command, cwd=repo, check=True)
        object_files.append(str(obj))
    command = [
        cxx_compiler, "-shared", "-fPIC", "-O3", "-std=c++17",
        "-I", str(repo / "f1tenth_mpc/include"),
        str(repo / "tools/racing/offline/controller/production_mpc_shim.cpp"),
        *object_files, "-lyaml-cpp", "-lm", "-o", str(output),
    ]
    subprocess.run(command, cwd=repo, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[4])
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--cxx-compiler", default="c++")
    args = parser.parse_args()
    build(args.repo.resolve(), args.output.resolve(),
          args.compiler, args.cxx_compiler)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
