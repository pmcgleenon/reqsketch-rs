# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/pmcgleenon/reqsketch-rs/compare/v0.1.1...v0.1.2) - 2026-04-18

### Other

- replace misleading sqrt(k/n) sigma with a 1/k tolerance
- wrap bare URL in the crate root in angle brackets
- add serde round-trip + rank/sorted_view equivalence coverage
- tighten correctness test margins to match observed accuracy
- expand rust workflow with fmt, clippy, MSRV, feature matrix, multi-OS
- cargo fmt + clippy cleanups for -D warnings readiness
- Make n explicitly u64 in test_weight_conservation loop
- Rewrite flaky sigma/accuracy tests with deterministic and multi-trial tests
- Replace O(n) linear duplicate scan with O(log n) partition_point in compute_weight
- Replace O(n) linear scan with O(log n) partition_point in rank_no_interpolation
- Simplify SortedView dedup: drop redundant last_mut guard
- Fix compilation error and clippy warning from previous fixes
- Fix potential usize underflow in compute_compaction_range
- Replace binary_search with partition_point in SortedView::quantile
- Add deduplication of equal items in SortedView construction
- Complete reset() by clearing num_retained and max_nom_size
- Fix section growth off-by-one in ensure_enough_sections
- Merge origin/main into fix/critical-correctness
- remove spurious compaction guard in compute_compaction_range
- correct error bound formulas for rank lower/upper bounds
- sort target compactor before merge_sorted in compress
- update num_retained and max_nom_size before compression check in merge
- move state increment before ensure_enough_sections
- use correct coin-flip logic in compact_into_fast
- sort entire buffer before compaction, not just compaction range

## [0.1.1](https://github.com/pmcgleenon/reqsketch-rs/compare/v0.1.0...v0.1.1) - 2025-12-20

### Fixed

- error bounds were too high for examples/req_rank_error.rs

### Other

- added useful links to readme
- Update rand_distr requirement from 0.4 to 0.5
- Update rand requirement from 0.8 to 0.9
- Update criterion requirement from 0.5 to 0.8
