# Release checklist

- [ ] Resolve or accurately disclose every item in RELEASE_BLOCKERS.md.
- [ ] Confirm no raw restricted or third-party subject-level data are included.
- [ ] Review sharing rights for aggregate logs and manuscript figures.
- [ ] Verify downloaded cache hashes/row counts and the DRSA MUSIC source/time-unit discrepancy, or state specific full-refit limitations.
- [x] Owner-selected CC BY 4.0 applied and metadata updated.
- [x] Author details confirmed by owner.
- [x] Owner waived editable supplement source; supplied PDF and wrapper retained.
- [ ] Review the E8 ordinal-rank versus standard Spearman definition.
- [ ] Complete clean installation test and document its environment.
- [ ] Run python scripts/reproduce_figures.py and visually review PDF and PNG outputs.
- [ ] Run python scripts/reproduce_tables.py and confirm frozen numerical outputs.
- [ ] Run python scripts/verify_repository.py; inspect results/verification.json.
- [ ] Complete the manuscript NUMBER_AUDIT.csv review; resolve unexplained differences.
- [ ] Fill the actual repository URL in README, CITATION.cff and .zenodo.json.
- [ ] Set the actual release date, complete RELEASE_APPROVAL.json, and run verification with --release.
- [ ] Initialise/push only this repository; inspect staged filenames first.
- [ ] Connect/enable the repository in production Zenodo BEFORE publishing the release.
- [ ] Create GitHub release v1.0.0 from the reviewed commit.
- [ ] Verify successful Zenodo archive, download/inspect it, and obtain its version DOI.
- [ ] Update README, CITATION.cff, .zenodo.json related metadata, references.bib, Data Availability and Code Availability with appropriate real identifiers.
- [ ] Keep v1.0.0 immutable. Create v1.0.1 only if a metadata-only archived update is actually needed; never recycle an earlier version DOI for a new version.

The exact order and commands are in GITHUB_ZENODO_STEPS.md.
