# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

First public release.

Find and reclaim idle GPU allocations in Slurm, observe-by-default.

- Core tool implemented and covered by tests.
- `make demo` (or equivalent) runs against a synthetic backend, no special
  hardware required.
- CI runs the test suite on every push.

This is a `0.x` release: the behaviour is tested and the safety properties are
asserted, but flags and metric names may still change before `1.0.0`.

[0.1.0]: https://github.com/Zhanyl-tech/gpu-reaper/releases/tag/v0.1.0
