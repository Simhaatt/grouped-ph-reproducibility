Current setup: repository created at https://github.com/Simhaatt/grouped-ph-reproducibility. The owner explicitly approved uploading this unreleased draft publicly. URL metadata has been filled. Public draft upload does not mean v1.0.0 has been released or the remaining review items are resolved. The steps below also document the full workflow; do not create a duplicate repository.

# Exactly what to create, and when

**Use only this prepared repository folder. Never upload the parent research directory or the private preservation archive. Nothing has been published yet.**

## 1. Now: finish the local publication decisions

Read RELEASE_BLOCKERS.md. The owner selected CC BY 4.0, confirmed author details and waived editable supplement source. These decisions and licence metadata are already recorded. Resolve or explicitly disclose the data-source, seed, correlation and missing-source limitations. The raw-cache acquisition mechanism is documented; review the DRSA MUSIC source discrepancy. Do a clean environment test. Review new figures and manuscript numerical claims. The existing frozen outputs must remain unchanged.

For documented review completion, create docs/RELEASE_APPROVAL.json with these keys set to true **only when reviewed**: licence_review, data_and_cache_review, seed_limitation_disclosed, correlation_definition_review, supplement_source_review, manuscript_review, clean_environment_test, author_metadata_review. This is a review record, not a way to bypass outstanding issues.

## 2. Then: create ONE GitHub repository

Sign in to GitHub → + → New repository. Name: **grouped-ph-reproducibility**. Description: “Reproducibility package for Grouped Proportional Hazards under Coarse Time: Information Loss and Baseline Estimation.” You may start private for coauthor review; make it public only after the release checklist passes and before enabling public archiving. Leave “Add README”, “Add .gitignore” and “Choose a license” unchecked: local files already exist.

Copy its HTTPS URL and replace YOUR_USERNAME in README.md and .zenodo.json. In CITATION.cff uncomment repository-code and insert that same URL. Set date-released only when the release date is known. Do not insert a fake DOI.

## 3. Push the prepared folder

Open PowerShell **inside grouped-ph-reproducibility**, not its parent. Replace YOUR_USERNAME below before running. Git must be installed.

```powershell
python scripts/reproduce_main_results.py
python scripts/verify_repository.py --release
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare reproducibility package v1.0.0"
git remote add origin https://github.com/Simhaatt/grouped-ph-reproducibility.git
git push -u origin main
```

For an approved draft upload, the ordinary frozen-output verifier must pass; the release gate remains required before publishing a tagged release. Inspect staged filenames before committing; no raw data, secrets, private archive, caches or checkpoints should appear. If git init already ran, do not initialise again; if origin already exists, inspect it before changing it. Do not push a tag or create a release yet.

## 4. BEFORE the first GitHub release: connect Zenodo

Use production [Zenodo](https://zenodo.org/), not its sandbox. Sign in (GitHub sign-in is convenient), link/authorise your GitHub account, open Zenodo's GitHub settings page, find this repository and enable its toggle. If the repository is missing, refresh/sync and verify that you are an administrator and the integration has access. For an organisation, its owner may need to approve the integration. Use a public repository for this public archive workflow.

Do **not** also create a separate manual upload for the same package. The GitHub integration will archive the release. Official instructions: [Enable a repository](https://help.zenodo.org/docs/github/enable-repository/).

## 5. Only after steps 1–4: publish GitHub release v1.0.0

Update the actual release date in CITATION.cff and CHANGELOG.md; commit and push that final metadata before tagging. Rerun release verification. On GitHub open repository → Releases → Draft a new release. Choose/create tag **v1.0.0** targeting the final main commit. Title: **v1.0.0 — Reproducibility package**. Paste docs/RELEASE_NOTES.md into the description. Leave “pre-release” unchecked. Review the draft, then click **Publish release**.

A branch push or Git tag alone is not this workflow's published release. There is no need to attach the original research ZIP or raw data; the tagged repository snapshot is the package. See [GitHub release management](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

## 6. After publishing: verify the Zenodo archive and copy the DOI

Return to Zenodo GitHub settings and wait for the archive to finish. Open the new record; if your integration shows a draft requiring review, complete its required metadata and publish it. Check version, creators, licence, source ZIP and repository link. Download the archived ZIP and inspect it. If archiving fails, read the error and fix metadata; do not create a duplicate manual record as a workaround without deciding to switch workflows. See [Archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/).

Copy the **version-specific DOI for v1.0.0** for the paper's exact reproducibility citation. The concept DOI represents the series of versions and is useful for a general project badge/link. Record both with their meanings. Do not confuse the software DOI with the manuscript's eventual journal DOI.

## 7. After the DOI exists: update citations without a release loop

On main, add the assigned v1.0.0 DOI to README and CITATION.cff. Add a BibTeX software entry in manuscript/references.bib and cite it in the manuscript's Code Availability section; keep Data Availability source/access restrictions accurate. Update .zenodo.json with accurate repository metadata and, if desired, a correctly typed relation to the existing archived release. Do not place the previous version's DOI in the top-level doi field of metadata intended for a new release.

Suggested Code Availability sentence after filling actual identifiers: “Code and frozen aggregate results for the analyses are archived at https://doi.org/ACTUAL_VERSION_DOI (version 1.0.0); development source is available at ACTUAL_GITHUB_URL. Third-party raw data are obtained from the original providers under their access conditions.” This paper-ready wording must be completed with real identifiers before submission.

Commit/push those citation changes. **Do not move or overwrite the v1.0.0 tag.** A citation update on main does not require another release. If you specifically need the post-DOI metadata snapshot archived, make **one v1.0.1 metadata-only release**, update its version/date in metadata, and let Zenodo assign that version its own DOI. Cite the version actually used. Do not repeatedly release solely to insert each new release's own DOI; use main for subsequent DOI links or a deliberate manual reserved-DOI workflow instead.

## 8. For later scientific changes

Retain v1.0.0 and its frozen outputs. Document changes in CHANGELOG.md, regenerate new outputs under new provenance, verify and publish a new version with an appropriate version increment. Zenodo creates a new version DOI. Never edit archived numerical values to match revised manuscript claims.

Metadata reference: [Zenodo JSON](https://help.zenodo.org/docs/github/describe-software/zenodo-json/). DOI reservation belongs to a separate deliberate upload workflow; it is unnecessary for the recommended integration sequence above.
