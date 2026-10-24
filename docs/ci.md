# GitHub Actions checks

The `CI` workflow runs on branch pushes, pull requests to `main`, and manual
dispatch. It uses GitHub-hosted Ubuntu 24.04 runners. Heavy package installation
and Docker builds happen on those runners, preserving this machine's 2 GiB disk
reserve and its running Compose stack.

## Project workflow

`Lint and tests` checks repository content and commit history, installs the pinned
`requirements/ci.lock` dependency closure, runs Ruff lint and format checks, and
runs the full synthetic pytest suite on Python 3.12. `pytest.ini` limits discovery
to `tests/`, keeping disposable runtime exercises outside the project suite.
The lock uses the same package versions as the training environment without
installing DVC, MLflow or dataset-download tools that the tests do not require.

`Docker build and Compose checks` runs after the test job succeeds. It validates
the Dockerfile, builds the real serving image and starts the production Compose
configuration with a small deterministic Keras fixture and a new PostgreSQL
volume. Red and blue synthetic PNGs exercise both labels and preprocessing.
Unsupported text, corrupt PNG and oversized uploads exercise rejection paths.
The check reads history to confirm all five requests were logged. It then removes
and recreates both containers and verifies the same records remain available.

The model fixture is explicitly labelled `synthetic`; it is not the champion.
The workflow uses no dataset images, held-out test set, DVC remote, DagsHub token
or production history. Fixture helpers refuse to overwrite existing settings or
a populated model directory. The run-specific database volume and test containers
are removed during cleanup, including after a failed check.

Test XML and a small container-check JSON are retained as private Actions
artifacts for one day. The image and fixture model are not uploaded as artifacts.
Runs have bounded job timeouts, and a newer push cancels an older run on the same
ref. All actions are official GitHub actions pinned to verified full commit SHAs;
the workflow token has read-only repository permissions and checkout does not
retain Git credentials.

## Run and inspect

Open the repository's Actions tab and select `CI`, or use the CLI:

```bash
gh workflow run ci.yml --ref main
gh run list --workflow ci.yml --limit 5
gh run view <run-id>
```

For local checks in the existing environment:

```bash
source .venv/bin/activate
python scripts/check_disk.py --minimum-free-gib 2
python scripts/check_repository.py --history
ruff check scripts visionguard tests docker wsgi.py gunicorn.conf.py
ruff format --check scripts visionguard tests docker wsgi.py gunicorn.conf.py
pytest -q
```

Use the guarded build from `docs/docker.md` for local image rebuilds. The CI
fixture commands are intended for a fresh disposable runner workspace.

## Secrets and deployment

These checks require no repository secrets. The temporary database password and
Flask key are generated at runtime and saved only in the runner's ignored `.env`
with mode 0600. They are not printed or uploaded. Existing local and DagsHub
credentials remain outside Git.

Future cloud deployment credentials belong in GitHub Settings, Secrets and
variables, Actions, or protected environment secrets. A workflow references them
through `secrets.NAME`; their values must not be embedded in YAML, source code,
image layers or logs. Cloud deployment remains a separate stage, using this
workflow's test and Docker validation gates.

## Two-version exercise

The private `Hanzala-Shadow/23L-0907_A2_Ops-ci-lab` repository contains only a small
arithmetic function, its test and a Python 3.11/3.12 matrix. The initial expectation
is intentionally wrong to observe a red run. The following commit fixes the
expectation to obtain a green run. This exercise has its own history and workflow.
Run links and results are recorded in `reports/continuous-integration/`.

References: [GitHub workflow security](https://docs.github.com/en/actions/reference/security/secure-use)
and [Docker test-before-push checks](https://docs.docker.com/build/ci/github-actions/test-before-push/).
