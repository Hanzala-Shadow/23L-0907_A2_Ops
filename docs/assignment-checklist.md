# Assignment completion audit

Reviewed against the seven-page assignment on 25 October 2026 (Asia/Karachi).
The report is `output/pdf/23L-0907_VisionGuard.pdf`. It records the graded source
revision, exact results, captions, deployment steps and assistance disclosure.

| ID | Completed implementation and evidence |
| --- | --- |
| 1.1 | Isolated `.venv`, exact package lock, CPU/GPU discovery and environment smoke checks; report environment transcript. |
| 1.2 | Custom per-file audit with class, bytes, dimensions, mode, format and SHA-256; `reports/data/`. One duplicate excluded from splits, zero corrupt files. |
| 1.3 | Class sample grid and three per-class histograms, with interpretations in the PDF and `reports/data/audit.md`. |
| 1.4 | Two-version toy CSV on the `dvc-lab` GitHub/DagsHub branch; older version restored after deleting its local file and cache. `reports/labs/dvc/`. Actual data/model pointers, DagsHub remote and verified full restore documented in `docs/data-versioning.md`. |
| 2.1 | Shared decoder, RGB white letterbox, `tf.data`, batching/prefetch, training-only augmentation; `docs/training.md`, augmented-batch figure. |
| 2.2 | Custom 28,145-parameter CNN, model summary, per-layer table and design explanations; `reports/training/cnn_baseline/`. |
| 2.3 | Frozen MobileNetV2 followed by last-block fine-tuning at a 100-fold lower rate, expected normalization and trainable-layer checks; `docs/transfer-learning.md`. |
| 2.4 | Seed 42, deterministic CPU configuration, early stopping, checkpoint and learning-rate reduction, curves and best epochs. |
| 2.5 | Validation-only model/threshold selection; one test inference pass for each of two locked models; metrics, confusion matrices and ROC curves in `reports/evaluation/`. |
| 2.6 | Ten saved misclassified test examples, individual visual hypotheses and sampling caveat; `reports/evaluation/README.md`. No extra test inference for reporting. |
| 2.7 | Four MNIST toy trials and eight complete VisionGuard factorial trials; DagsHub runs, tables and validation-overfitting discussion; `docs/tuning*.md`. |
| 3.1 | Two tiny MNIST local MLflow runs with distinct learning rates, epoch metrics, plots and models; genuine comparison screenshots; `reports/labs/mlflow/`. |
| 3.2 | Private GitHub-connected DagsHub project, environment-only authentication and successful repeat of the toy run; `.env` ignored. |
| 3.3 | Manual CNN callback and MobileNet TensorFlow autologging, provenance tags, epoch metrics and attached final test metrics. |
| 3.4 | Models, plots, split lists, package lists and evaluation artifacts on MLflow, with artifact-browser screenshot. |
| 3.5 | Eleven candidate results and a genuine six-run DagsHub comparison; validation-loss justification retained. |
| 3.6 | Two registry versions, champion alias on version 2, separate alias-load prediction script and receipt; `reports/registry/`. |
| 4.1 | Native model loaded once per worker; shared preprocessing and known validation-image checks. Current cloud interpreter loads once per warm instance; native image retained. |
| 4.2 | Upload/result/history/health routes; wrong type, corrupt image, byte/pixel limits and safe filenames; three saved rejection screenshots. |
| 4.3 | HTML/CSS upload, confidence/result/caveat and PostgreSQL-backed history menu. |
| 4.4 | Environment configuration, saved threshold, debug off, recorded Flask-server check and current Gunicorn deployment; health screenshot. |
| 5 | PostgreSQL records timestamps, decisions, confidence, model provenance and safe failure metadata; persistence and original-row comparisons. |
| 6.1 | Official-image warm-up with port, volume, logs, shell, lifecycle, listing and cleanup. The exercise was completed retrospectively; it is not presented as predating the project. `reports/labs/docker/`. |
| 6.2 | Dependency-first slim Dockerfiles, non-root user, health checks, allowlisted build contexts and CPU/native versus remote-client explanation. |
| 6.3 | Web/database services, service-name networking, readiness, named volume, ignored secrets and read-only model mount. PostgreSQL has a localhost-only administration mapping and no public bind. |
| 6.4 | Same-command image comparison: native 1.72 GB to remote client 395 MB; layer histories and optimization tradeoff in `reports/labs/docker/verification.json`. |
| 6.5 | Isolated missing-model failure diagnosed from logs, corrected read-only mount and healthy recovery; start/stop/restart/logs/shell actions for both services. Production data untouched. |
| 6.6 | Authorized private toy CI repository, red then green Python 3.11/3.12 matrix; main lint/tests/Docker/Compose checks and README badge. |
| 7 | Vercel Hobby model API, step-by-step deployment explanation, public URL, official Postman CLI run from GitHub's separate network: six requests, 13 passing assertions. `reports/labs/postman/`. |
| 8 | Local Flask to Vercel, multiple generated inputs, saved results/history, timeout/network handling and pinned model contract; `reports/integration/`. |

## Submission notes

- The student confirmed the cover name as Syed Muhammad Hanzala and no stated page limit.
- The student will grant grading access to both private repositories separately.
- The API key belongs in a private grader channel or local Postman environment,
  never the PDF, a URL, the collection or committed files.
- Task 7's direct instruction and Task 8 specify localhost Flask calling the
  cloud model API. The implemented layout follows those instructions.
- The Vercel resource name is normalized for DNS as `23l-0907-a2-ops-api`.
- Transcript screenshots explicitly identify recorded command output; UI
  screenshots are genuine captures. No terminal window or cloud console is fabricated.
- The optional Grad-CAM bonus is not attempted. Core requirements are retained
  independently of that bonus.
- The final audit uses saved evaluation outputs. It does not retrain the champion,
  change its threshold, or rerun the held-out test set.

## Rebuild the report

Run `scripts/build_submission_report.py` using a Python runtime with ReportLab
and Pillow. The builder reads versioned results and screenshots and writes the
named PDF. It performs no model inference or network calls. The source revision
on the cover identifies code/evidence; a following report-only commit contains
the PDF so its hash does not refer to itself.

The DVC lab uses a fresh ignored folder and refuses to overwrite an existing lab.
`scripts/lab_mlflow.py` uses only MNIST training-partition examples. Its
`--remote-only` mode repeats the cloud demonstration without duplicating local
runs. `scripts/lab_docker.py` checks for naming collisions, creates its own volume,
and removes only its lab resources. Postman uses generated images and never the
dataset or local history. Do not rerun labs merely to regenerate the report.
