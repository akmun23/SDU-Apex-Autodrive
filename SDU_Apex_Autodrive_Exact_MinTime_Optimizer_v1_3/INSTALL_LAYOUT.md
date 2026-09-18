# Installation layout

The runner supports both of these layouts.

## A. Overlay into the repository (recommended)

Unzip/copy the bundle so that its `f1tenth_planning/` directory merges with the repository's existing `f1tenth_planning/` directory.

Run from anywhere:

```bash
python3 /path/to/SDU-Apex-Autodrive/f1tenth_planning/scripts/run_autodrive_mintime.py --self-test
```

## B. Keep the bundle as a nested directory

This is also supported, for example:

```text
SDU-Apex-Autodrive/
├── f1tenth_mpc/
├── f1tenth_planning/
└── SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/
    └── f1tenth_planning/
        ├── autodrive_mintime/
        ├── config/
        └── scripts/
```

Then this works directly:

```bash
cd SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/f1tenth_planning/scripts
python3 run_autodrive_mintime.py --self-test
python3 run_autodrive_mintime.py
```

The runner distinguishes the optimizer **tool root** from the actual **repository root** automatically.

## Checkpoint and failure recovery

Each converged continuation level is saved below the configured output path in
`checkpoints/`. Failed IPOPT refinement iterates are also saved there for
analysis. A fine-mesh failure therefore no longer leaves an empty output tree.
