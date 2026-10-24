# CI verification

The [project workflow](https://github.com/Hanzala-Shadow/23L-0907_A2_Ops/actions/runs/40117744561)
passed both `Lint and tests` and `Docker build and Compose checks` for commit
`a7ffaeb4aba94112d2f38ab3cf0a219300917dd1`.

The downloaded JUnit artifact confirms 77 tests, zero failures, zero errors and
zero skipped tests. Ruff lint and formatting, dependency checks and repository
content checks passed. The Docker job built the serving image and ran the actual
Compose configuration with synthetic fixtures and an isolated PostgreSQL volume.
Its artifact confirms two classifications, three rejected uploads, five history
rows and persistence after removing and recreating both containers.

The first hosted integration run exposed an HTTP-client issue: the application
returned 413 before the oversized body finished sending, and urllib lost the
response with a broken pipe. The smoke client now uses the pinned Requests
library. A direct local early-response regression check and the hosted upload
check both passed. A YAML formatting error in the added install step was also
fixed before the successful run. The application's upload limit was unchanged.

## Two-version exercise

The approved private [CI lab](https://github.com/Hanzala-Shadow/23L-0907_A2_Ops-ci-lab)
has a [red run](https://github.com/Hanzala-Shadow/23L-0907_A2_Ops-ci-lab/actions/runs/40116974037)
and a [green run](https://github.com/Hanzala-Shadow/23L-0907_A2_Ops-ci-lab/actions/runs/40117054547).
Both Python 3.11 and 3.12 failed the intentional `add(2, 2) == 5` assertion, then
passed after changing the expected result to 4. Job-log excerpts record the real
failure and success for each interpreter in `lab-runs.json`.

## Local state and evidence

The local 77-test suite passed, including model loading, threshold enforcement,
preprocessing, upload validation and history operations. Test scratch files were
removed after collecting the XML results. No local Docker image build or package
installation was needed. Free space remained above the 2 GiB reserve, with about
3.04 GiB free after the checks.

The existing Docker app remained healthy. All 11 PostgreSQL history records and
the champion checkpoint hash were preserved. DVC reports the data and pipelines
up to date. CI used only synthetic models and images and did not evaluate the
held-out test set.

- `project-run.json`: successful hosted jobs and verified artifact results.
- `lab-runs.json`: separate Python matrix red/green evidence.
- `local-checks.json`: local test counts and early-413 regression check.
- `local-state.json`: live app, original history count and champion checksum.

The workflow and operational details are in `docs/ci.md`. The README status badge
links directly to Actions. Workflow actions are pinned, token permissions are
read-only, and private reports expire after one day. Deployment credentials can
be configured through repository or environment secrets for the later cloud
deployment stage.
