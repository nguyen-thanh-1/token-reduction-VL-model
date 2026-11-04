# Architecture workspace

This directory keeps editable network diagrams and their exported images under
version control. Each architecture or experiment family must use its own
subdirectory so baseline and pruning designs are never mixed.

```text
architecture/
├── model-goc-baseline/
│   ├── Qwen3-VL-2B-Instruct-architecture.drawio
│   ├── README.md
│   └── exports/                      # PNG, SVG, or PDF exports
├── dsfold-proposal/                  # proposed intervention, not implemented
│   ├── Qwen3-VL-2B-Instruct-DSFold-proposal.drawio
│   ├── README.md
│   └── exports/
└── pp3-dsfold-scorer/                # PP3: DSFold + learned score head
    ├── Qwen3-VL-2B-Instruct-PP3-scorer.drawio   # full-model copy, head 3a
    ├── PP3-3d-attention-scorer.drawio           # focused diagram, current head 3d
    ├── README.md
    └── exports/                                 # PP3-3d-attention-scorer.png / .svg
```

The baseline diagram is a single page that exposes the complete execution and
token flow, with detail panels for the vision block, the decoder layer, and a
token/tensor ledger.

For future methods, create sibling directories such as `random-k/`,
`uniform-k/`, or `adaptive-pruning/`. Keep the editable `.drawio` source beside
a short README that records the model revision and assumptions.
