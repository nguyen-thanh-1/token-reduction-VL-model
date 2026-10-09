# Architecture workspace

This directory keeps editable network diagrams and their exported images under
version control. Each architecture or experiment family must use its own
subdirectory so baseline and pruning designs are never mixed.

```text
architecture/
└── model-goc-baseline/
    ├── qwen3-vl-2b-instruct-baseline.drawio
    ├── qwen3-vl-2b-instruct-single-page-full-architecture.drawio
    ├── README.md
    └── exports/                      # PNG, SVG, or PDF exports
```

The baseline directory intentionally keeps both forms: the multi-page source
is easier to read module by module, while the single-page source exposes the
complete execution and token flow without page transitions.

For future methods, create sibling directories such as `random-k/`,
`uniform-k/`, or `adaptive-pruning/`. Keep the editable `.drawio` source beside
a short README that records the model revision and assumptions.
