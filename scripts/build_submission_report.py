"""Build a concise assignment report from saved results; perform no inference."""

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

from PIL import Image as PILImage
from PIL import ImageChops
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 528
EVIDENCE = "reports/submission/evidence/"


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    if shutil.disk_usage(ROOT).free < 2.125 * 1024**3:
        raise RuntimeError("Keep the 2 GiB reserve plus report headroom")
    for name, filename in [("Body", "DejaVuSans"), ("Bold", "DejaVuSans-Bold")]:
        pdfmetrics.registerFont(
            TTFont(name, f"/usr/share/fonts/truetype/dejavu/{filename}.ttf")
        )
    ink = colors.HexColor("#16394c")
    styles = {
        "body": ParagraphStyle(
            "body", fontName="Body", fontSize=9.5, leading=13, spaceAfter=7
        ),
        "title": ParagraphStyle(
            "title",
            fontName="Bold",
            fontSize=20,
            leading=25,
            textColor=ink,
            spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="Bold",
            fontSize=11,
            leading=15,
            textColor=ink,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Body",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#43545d"),
            spaceBefore=4,
            spaceAfter=7,
        ),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=8, leading=10.5),
        "small": ParagraphStyle(
            "small", fontName="Body", fontSize=8, leading=11, spaceAfter=6
        ),
    }
    story = []
    figures = 0
    now = datetime.now(ZoneInfo("Asia/Karachi"))
    # Preserve the graded implementation revision when only the report changes.
    commit = read("reports/submission/verification.json")["source_commit"]
    assets = ROOT / "reports/training-runtime/submission/concise-assets"
    assets.mkdir(parents=True, exist_ok=True)

    def p(text, style="body"):
        story.append(Paragraph(text, styles[style]))

    def h(text):
        p(text, "heading")

    def page(title):
        if story:
            story.append(PageBreak())
        p(title, "title")

    def table(headers, rows, widths=None, compact=False):
        data = [
            [Paragraph(escape(str(c)), styles["cell"]) for c in row]
            for row in [headers, *rows]
        ]
        result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8ec")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f4f6f7")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 3 if compact else 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 if compact else 5),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#90a5ad")),
                ]
            )
        )
        story.extend([result, Spacer(1, 7)])

    def panel(path, caption, width=WIDTH, height=190):
        nonlocal figures
        figures += 1
        source = ROOT / path
        with PILImage.open(source) as original:
            im = original.convert("RGB")
        if source.stem in {"app-health", "cloud-health"}:
            im = im.crop((0, 0, im.width - 16, im.height))
            bounds = ImageChops.difference(
                im, PILImage.new("RGB", im.size, "white")
            ).getbbox()
            if bounds:
                im = im.crop((0, 0, im.width, min(im.height, bounds[3] + 15)))
        scale = min(width / im.width, height / im.height)
        w, ht = im.width * scale, im.height * scale
        im.thumbnail((round(w * 3), round(ht * 3)))
        target = assets / f"figure-{figures}.jpg"
        im.save(target, quality=92, subsampling=0, optimize=True)
        im.close()
        return [
            Image(str(target), width=w, height=ht),
            Paragraph(f"Figure {figures}. {escape(caption)}", styles["caption"]),
        ]

    def figure(path, caption, height=190):
        story.append(KeepTogether(panel(path, caption, height=height)))

    def row(items, height=140):
        gap = 12
        cell_width = (WIDTH - gap * (len(items) - 1)) / len(items)
        cells = []
        for i, (path, caption) in enumerate(items):
            if i:
                cells.append("")
            cells.append(panel(path, caption, width=cell_width, height=height))
        widths = [cell_width if i % 2 == 0 else gap for i in range(len(cells))]
        result = Table([cells], colWidths=widths, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(result)

    def shot(name):
        return EVIDENCE + name + ".png"

    # 1: front matter and a short outcome, without a separate reading-guide chapter.
    page("VisionGuard")
    p("AI-generated and real image classification", "heading")
    p("MLOps Assignment 2", "heading")
    story.append(Spacer(1, 14))
    table(
        ["Submission", "Details"],
        [
            ["Name", "Syed Muhammad Hanzala"],
            ["Roll number / project", "23L-0907 / 23L-0907_A2_Ops"],
            ["Report date", now.strftime("%d %B %Y")],
            ["Graded implementation", commit],
            ["GitHub", "https://github.com/Hanzala-Shadow/23L-0907_A2_Ops"],
            ["DagsHub", "https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops"],
            ["Live model API", "https://23l-0907-a2-ops-api.vercel.app/predict"],
            ["Public health check", "https://23l-0907-a2-ops-api.vercel.app/health"],
            ["Local web app", "http://127.0.0.1:5000/"],
        ],
        [130, 398],
    )
    h("Project outcome")
    p(
        "VisionGuard audits and versions the dataset, compares a custom CNN with MobileNetV2, "
        "tracks experiments on DagsHub, and serves the selected model through a cloud API. "
        "The local Flask interface sends uploads to that API and records results in PostgreSQL. "
        "The tuned model achieved <b>72.11% test accuracy and 0.8161 ROC-AUC</b>."
    )
    h("Report scope")
    p(
        "Tasks 1-8 are covered with the main decisions, results and essential evidence. "
        "Detailed commands, full screenshots and machine-readable receipts remain in the repository; "
        "docs/assignment-checklist.md maps every requirement to its evidence. The optional Grad-CAM "
        "bonus was not attempted."
    )
    h("Assistance disclosure")
    p(
        "OpenAI Codex assisted with setup, implementation, debugging, experiments, deployment and "
        "report drafting. Results are backed by saved scripts, metrics and execution receipts."
    )
    p(
        "Both repositories are private; grading access will be shared separately. The prediction "
        "API requires a key, which must be shared privately. The implementation revision above "
        "is unchanged by this shortened report.",
        "small",
    )

    # 2: compact audit and genuine environment/DVC evidence.
    page("Task 1 - Data and versioning")
    h("1.1 Environment")
    p(
        "An isolated .venv uses Python 3.12.3, TensorFlow CPU 2.19.1, Keras 3.15.1, "
        "Keras Tuner 1.4.8, MLflow 2.22.5 and DVC 3.67.1. TensorFlow detected no GPU. "
        "Pinned requirements, bounded CPU threads and pre-operation disk checks keep execution "
        "reproducible within the 2 GiB free-space reserve (Figure 1)."
    )
    h("1.2 Dataset audit and splits")
    audit = read("reports/data/audit_summary.json")
    audit_rows = []
    for label, item in audit["classes"].items():
        audit_rows.append(
            [
                label,
                item["total_files"],
                " / ".join(str(item["width"][k]) for k in ["min", "median", "max"]),
                " / ".join(str(item["height"][k]) for k in ["min", "median", "max"]),
                " / ".join(
                    str(item["file_size_bytes"][k]) for k in ["min", "median", "max"]
                ),
            ]
        )
    table(
        [
            "Class",
            "Files",
            "Width min/med/max",
            "Height min/med/max",
            "Bytes min/med/max",
        ],
        audit_rows,
        [76, 32, 112, 112, 196],
        compact=True,
    )
    p(
        "The custom walker records class, bytes, dimensions, colour mode, format and SHA-256. "
        "All 975 files decode: <b>0 corrupt files, 1 exact duplicate pair</b> (image33.jpg/image34.jpg). "
        "Raw files are preserved; image34.jpg is excluded from the 974-image split inventory."
    )
    table(
        ["Split", "AI-generated", "Real-labeled", "Total"],
        [["Train", 376, 305, 681], ["Validation", 81, 65, 146], ["Test", 81, 66, 147]],
        [132] * 4,
        compact=True,
    )
    p(
        "Seed-42 hash ranking produces stratified 70/15/15 lists. Validation checks confirm no path "
        "or hash overlap and identical lists on reproduction. Exact hashing does not detect near-duplicates."
    )
    h("1.4 DVC storage and toy restore")
    p(
        "The toy CSV has two committed versions on the dvc-lab branch. After checking out the older "
        "version and deleting both its local data and cache, dvc pull restored the original bytes "
        "(Figure 2). Its pointer holds the checksum, size and path; large content stays outside Git. "
        "Project data, splits and model bundles use DagsHub DVC storage; credentials remain ignored."
    )
    row(
        [
            (shot("environment"), "Recorded versions and GPU check."),
            (shot("dvc-lab"), "Two toy commits and verified old-version restore."),
        ],
        height=139,
    )

    # 3: every required EDA plot, with brief interpretations.
    page("Task 1 - Visual exploration")
    h("1.3 Samples and distributions")
    figure(
        "reports/data/sample_grid.png",
        "Seeded examples from both folder labels.",
        height=210,
    )
    p(
        "The grid includes photographs, artwork, composites and web graphics. Folder labels define "
        "the task target, but image origin was not independently verified; this diversity can confuse the model."
    )
    row(
        [
            ("reports/data/width_histogram.png", "Image widths by class."),
            ("reports/data/height_histogram.png", "Image heights by class."),
        ],
        height=135,
    )
    p(
        "<b>Widths:</b> Both classes span small images to several thousand pixels, with common generation "
        "sizes visible in the AI class. Resizing bounds CPU work. <b>Heights:</b> Portrait and landscape "
        "shapes vary substantially. Letterboxing preserves aspect ratio instead of stretching subjects."
    )
    figure(
        "reports/data/file_size_bytes_histogram.png",
        "Per-class file sizes, showing a long tail.",
        height=116,
    )
    p(
        "File sizes depend on compression and content as well as dimensions. Byte limits alone cannot "
        "bound decoded memory, so serving also checks pixel count."
    )

    # 4: full per-layer table plus the required augmented-batch figure.
    page("Task 2 - Pipeline and CNN")
    h("2.1 Input pipeline")
    p(
        "Split CSVs feed tf.data. The shared decoder corrects EXIF orientation, converts to RGB, "
        "composites transparency on white and letterboxes to 128 x 128 with bilinear resizing. "
        "This preserves aspect ratio and limits CPU cost. Decoded uint8 tensors are cached in RAM "
        "(about 39 MiB), training is shuffled with seed 42, then batched by 16 and prefetched one batch."
    )
    p(
        "Training alone uses horizontal flips, rotation up to 18 degrees and zoom up to 10%. "
        "Validation and inference use the unaugmented decoder; normalization is saved inside the model."
    )
    figure(
        "reports/training/cnn_baseline/augmented_batch.png",
        "Original and augmented training examples.",
        height=108,
    )
    h("2.2 Custom CNN architecture")
    with (ROOT / "reports/training/cnn_baseline/layers.csv").open() as handle:
        layers = list(csv.DictReader(handle))
    table(
        ["Layer", "Type", "Output shape (batch omitted)", "Parameters"],
        [
            [
                r["layer"],
                r["type"],
                r["output_shape"].replace("None, ", ""),
                r["parameters"],
            ]
            for r in layers
        ],
        [125, 143, 185, 75],
        compact=True,
    )
    p(
        "Convolutions learn local features; BatchNorm stabilizes activations; ReLU adds nonlinearity; "
        "pooling reduces spatial cost. Global averaging limits parameters, Dense(64) combines features, "
        "dropout 0.3 regularizes, and a sigmoid gives AI probability. Total parameters: <b>28,145</b>.",
        "small",
    )

    # 5: summary screenshot and two transfer learning curves.
    page("Task 2 - Transfer learning")
    p(
        "The CNN uses pixel/255 normalization, binary cross-entropy and Adam at 0.001. "
        "Accuracy, precision, recall and ROC-AUC expose different error types. Its recorded model "
        "summary confirms the layer table on the preceding page (Figure 8)."
    )
    figure(
        shot("model-summary"),
        "Recorded CNN summary excerpt; the full output is saved in model_summary.txt.",
        height=175,
    )
    h("2.3 Frozen backbone and last-block fine-tuning")
    p(
        "ImageNet MobileNetV2 is loaded without its classifier, followed by global averaging, "
        "Dense(64), dropout 0.3 and a sigmoid. Its saved normalization is pixel/127.5 - 1, "
        "matching the pretrained backbone. Applying CNN-style normalization would change its input distribution."
    )
    table(
        ["Stage", "Trainable layers / parameters", "Adam rate"],
        [
            [
                "Frozen",
                "New dense classifier only; 82,049 trainable parameters",
                "0.001",
            ],
            [
                "Fine-tuned",
                "Head + block_16_expand, block_16_depthwise, block_16_project; 551,489 parameters",
                "0.00001",
            ],
        ],
        [73, 370, 85],
        compact=True,
    )
    p(
        "Every BatchNorm layer, earlier block and Conv_1 remain frozen. The second stage starts from "
        "the best frozen checkpoint and recompiles the optimizer. Inference-mode backbone statistics "
        "avoid changing BatchNorm estimates; saved weight checks confirm the intended boundaries."
    )
    row(
        [
            (
                "reports/training/mobilenetv2_frozen/training_curves.png",
                "Frozen-head curves; best validation epoch marked.",
            ),
            (
                "reports/training/mobilenetv2_finetuned/training_curves.png",
                "Last-block fine-tuning did not improve validation loss.",
            ),
        ],
        height=115,
    )

    # 6: no separate multi-page tuner narrative.
    page("Task 2 - Training and tuning")
    h("2.4 Training discipline")
    p(
        "Seed 42 fixes Python, NumPy and TensorFlow randomness. Deterministic operations, bounded "
        "threads, early stopping, ReduceLROnPlateau and checkpoints on minimum validation loss are used. "
        "Best epochs: CNN <b>1/6</b>, frozen MobileNet <b>3/6</b>, fine-tuned MobileNet <b>2/5</b>. "
        "Only saved best checkpoints are compared; exact cross-hardware equivalence is not assumed."
    )
    figure(
        "reports/training/cnn_baseline/training_curves.png",
        "CNN loss and accuracy with best epoch 1 marked.",
        height=105,
    )
    h("2.7 Keras Tuner: toy lab and eight project trials")
    p(
        "The MNIST toy uses 6,000 training and 1,000 validation examples from its training partition, "
        "with RandomSearch over learning rate and hidden units. Four three-epoch trials are logged to DagsHub."
    )
    toy = read("reports/training/toy_tuner/summary.json")
    table(
        ["Toy trial", "Learning rate", "Units", "Val. loss", "Val. accuracy"],
        [
            [
                r["trial_id"],
                r["learning_rate"],
                r["units"],
                f"{r['validation_loss']:.4f}",
                f"{r['validation_accuracy']:.4f}",
            ]
            for r in toy["trials"]
        ],
        [78, 115, 75, 130, 130],
        compact=True,
    )
    trials = []
    for i in range(8):
        hp = read(f"reports/training/keras_tuner/trial_{i}/configuration.json")[
            "hyperparameters"
        ]
        metrics = read(f"reports/training/keras_tuner/trial_{i}/metrics.json")
        trials.append(
            [
                i,
                hp["learning_rate"],
                hp["dropout"],
                hp["unfrozen_layers"],
                f"{metrics['validation_loss']:.5f}",
                metrics["best_epoch"],
            ]
        )
    table(
        [
            "Trial",
            "Learning rate",
            "Dropout",
            "Unfrozen convs",
            "Val. loss",
            "Best epoch",
        ],
        trials,
        [42, 104, 75, 115, 102, 90],
        compact=True,
    )
    p(
        "All eight project trials start from the same frozen checkpoint. <b>Trial 7 wins:</b> rate 0.0001, "
        "dropout 0.5 and three unfrozen convolutions, validation loss 0.48514. Higher dropout has the "
        "largest average effect (loss lower by 0.02322). Repeated selection can overfit validation; "
        "the test set was never used for tuning."
    )

    # 7: both models, all required plots, locked threshold explanation.
    page("Task 2 - Final evaluation")
    h("2.5 Model selection and threshold")
    p(
        "Minimum validation loss selects trial 7 from eleven candidates. Threshold <b>0.6196606457</b> "
        "minimizes 2 x false positives + false negatives on validation: calling a real image fake is "
        "assigned twice the cost of missing an AI image. This is a product assumption. The CNN threshold "
        "is 0.4725604951. Both choices were locked before the single test evaluation per model."
    )
    results = []
    for role in ["cnn_baseline", "champion"]:
        metrics = read(f"reports/evaluation/{role}/metrics.json")
        results.append(
            [
                role,
                *[
                    f"{metrics[k]:.4f}"
                    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                ],
            ]
        )
    table(
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        results,
        [143, 77, 77, 77, 77, 77],
        compact=True,
    )
    p(
        "Test contains 147 images (81 AI, 66 real). The champion has TN=51, FP=15, FN=26, TP=55; "
        "the CNN has TN=56, FP=10, FN=78, TP=3. Figures 12-15 use saved test probabilities; "
        "hash-bound receipts prevent another inference pass."
    )
    for role, label in [("cnn_baseline", "CNN"), ("champion", "Champion")]:
        row(
            [
                (
                    f"reports/evaluation/{role}/confusion_matrix.png",
                    label + " confusion matrix.",
                ),
                (f"reports/evaluation/{role}/roc.png", label + " ROC curve."),
            ],
            height=154,
        )
    p(
        "Champion accuracy fell from 80.82% on validation to 72.11% on test. Small, diverse data and "
        "repeated validation selection limit generalization. The weak CNN result is retained, and "
        "test performance did not trigger further tuning. Predictions do not prove image origin."
    )

    # 8: the ten-image grid and one short hypothesis for each.
    page("Task 2 - Error analysis")
    h("2.6 Ten misclassified test images")
    figure(
        "reports/evaluation/misclassified_examples.png",
        "First ten errors in fixed test-manifest order, selected from the saved evaluation.",
        height=300,
    )
    p(
        "All ten are false negatives because AI-labeled paths appear first in the manifest. They are "
        "not representative of every error type; the full test also has 15 false positives."
    )
    hypotheses = [
        "Multi-photo web layout may favor familiar photographic cues.",
        "Nebula-like painterly texture overlaps with real-labeled artwork.",
        "Photographic framing may dominate the small generated artwork.",
        "Fur and garden textures look familiar; 0.617 lies just below the cutoff.",
        "Small artworks in a product graphic lose detail after resizing.",
        "Infographic text and cutouts differ from an isolated generated image.",
        "Grainy monochrome portrait may resemble a real photograph.",
        "Saturated landscape shares scene structure with real-labeled artwork.",
        "Plausible mountain scenery may outweigh synthesis artifacts.",
        "Banner text and small portrait dominate; probability 0.548 misses the cutoff.",
    ]
    table(
        ["Example", "Visual hypothesis"],
        [[i + 1, t] for i, t in enumerate(hypotheses)],
        [58, 470],
        compact=True,
    )
    p(
        "These are visual hypotheses, not causal explanations. They suggest future data review "
        "without changing the locked evaluation.",
        "small",
    )

    # 9: local/remote toy and both logging modes in one page.
    page("Task 3 - MLflow and DagsHub")
    h("3.1 Local tool lab; 3.2 DagsHub setup")
    p(
        "A 16-unit MNIST network trains on 1,200 examples with 200 validation examples for two epochs. "
        "Learning rates 0.001 and 0.01 give validation accuracy 0.54 and 0.80, respectively. "
        "Both local runs log parameters, epoch metrics, model and loss plot. Repeating the 0.001 run "
        "on DagsHub reproduces 0.54 accuracy (Figures 17-18)."
    )
    p(
        "An <b>experiment</b> groups attempts; a <b>run</b> records one attempt; <b>parameters</b> "
        "describe configuration; <b>metrics</b> measure performance; <b>artifacts</b> are saved files; "
        "<b>tags</b> record context. Private DagsHub mirrors GitHub and provides the project MLflow "
        "tracking URI. Credentials are read from environment variables in ignored .env files."
    )
    row(
        [
            (shot("mlflow-comparison-metrics"), "Local MLflow toy comparison."),
            (shot("dagshub-toy"), "Verified DagsHub toy run."),
        ],
        height=142,
    )
    h("3.3 Manual logging and TensorFlow autologging")
    p(
        "The CNN callback manually logs epoch loss, accuracy, precision, recall, AUC and learning rate. "
        "MobileNet uses TensorFlow autologging for fit configuration and epoch metrics. Both explicitly "
        "add seed, Git revision, DVC hash, split lists and the best checkpoint. Auto-upload of the final "
        "model is disabled so it cannot replace the best checkpoint; tuner trials log explicitly."
    )
    row(
        [
            (shot("dagshub-manual-run"), "CNN manual logging."),
            (shot("dagshub-autolog-run"), "Frozen MobileNet autologging."),
        ],
        height=142,
    )
    p(
        "A local/remote tracking-address mix-up in the toy lab was corrected by capturing the cloud URI "
        "before switching to local storage. Only the independently verified remote run is claimed here.",
        "small",
    )

    # 10: comparison table and the essential artifact/registry evidence.
    page("Task 3 - Comparison and registry")
    h("3.4 Artifacts; 3.5 Meaningful comparison")
    p(
        "Runs retain model, configuration, packages, split lists, layer summary, training curves and "
        "validation outputs. Final evaluation stores both models' test plots and saved predictions, "
        "with final metrics attached to the originating training runs. The table compares eleven "
        "candidates; six distinct tuning runs are also shown in Figure 22."
    )
    candidates = [
        *read("reports/training/model_comparison.json")["models"],
        *[
            {
                "model": f"tuner_{i}",
                **read(f"reports/training/keras_tuner/trial_{i}/metrics.json"),
            }
            for i in range(8)
        ],
    ]
    table(
        ["Candidate", "Val. loss", "Accuracy at 0.5", "Val. ROC-AUC"],
        [
            [
                r["model"],
                f"{r['validation_loss']:.5f}",
                f"{r['validation_accuracy']:.4f}",
                f"{r['validation_roc_auc']:.4f}",
            ]
            for r in candidates
        ],
        [210, 96, 116, 106],
        compact=True,
    )
    row(
        [
            (shot("dagshub-artifacts"), "Saved model and reporting artifacts."),
            (shot("dagshub-comparison"), "Six runs with distinct tuning parameters."),
        ],
        height=116,
    )
    h("3.6 Registry and independent alias load")
    p(
        "Registry <b>23L-0907_A2_Ops</b> contains version 1 (frozen MobileNet) and version 2 "
        "(tuned MobileNet); alias <b>champion</b> points to version 2. The separate script "
        "scripts/predict_registry.py loads models:/23L-0907_A2_Ops@champion and predicts a validation "
        "image: probability 0.9492709637 at saved threshold 0.6196606457 (Figures 23-24)."
    )
    row(
        [
            (shot("dagshub-registry"), "Two versions and champion alias."),
            (shot("registry-load"), "Recorded independent alias-load prediction."),
        ],
        height=112,
    )

    # 11: the app and PostgreSQL form one coherent workflow.
    page("Tasks 4-5 - Flask and PostgreSQL")
    h("4.1 Model loading; 4.2 Routes and guards")
    p(
        "Native serving loads and validates the champion once per worker and uses the training decoder. "
        "The retained native image and model remain available. Current Flask uses one remote client per "
        "worker; the cloud interpreter loads once per warm instance. Two original validation-image checks "
        "and all 146 validation export-parity decisions are recorded; no test data was reused."
    )
    p(
        "Routes provide upload, prediction, saved result, paginated history and health. Upload checks "
        "cover form token, one file, sanitized name, type, decoding, 4 MiB bytes and 4 million pixels. "
        "JPEG, PNG and WebP stay in memory; only metadata and outcomes are stored."
    )
    h("4.3 Interface; 4.4 Configuration")
    p(
        "HTML/CSS presents upload, verdict/confidence and history (Figures 25-28). Environment variables "
        "supply database, model policy, server secrets and cloud settings. Debug is off; the development "
        "Flask server was checked, and Compose runs Gunicorn with one worker and two threads. "
        "A configured threshold must match the saved policy; the API key stays server-side."
    )
    row(
        [
            (shot("app-upload"), "Local upload page."),
            (shot("app-result"), "Saved verdict, confidence and version."),
        ],
        height=136,
    )
    h("Task 5 - Persistent database trail")
    p(
        "PostgreSQL records UTC time, safe filename, image hash/size, status, label, probability, "
        "confidence, threshold, latency and model version/identity. Rejections and cloud failures "
        "have safe error fields and no verdict. SQLAlchemy transactions back the history menu; "
        "an external named volume preserves rows across recreation. All <b>11 original rows</b> "
        "survived; eight integration checks brought the total to <b>19</b>."
    )
    row(
        [
            (shot("app-history"), "PostgreSQL-backed history."),
            (shot("app-health"), "Healthy database, remote model and debug off."),
        ],
        height=125,
    )

    # 12: Docker facts and three concise, genuine evidence panels.
    page("Task 6 - Docker Compose")
    h("6.1 Ready-made image lab")
    p(
        "An isolated postgres:17-alpine lab exercised run, ports, volume, logs, shell, listing, stop, "
        "start, restart and cleanup. An image is a template; a container is its instance; a layer is "
        "a cached build change; a volume outlives a container; port mapping forwards a host port. "
        "Only lab resources were removed. This warm-up and the DVC toy were completed during final audit."
    )
    h("6.2 Image design; 6.3 Compose configuration")
    table(
        ["Component", "Configuration and reason"],
        [
            [
                "Images",
                "Pinned Python slim; dependencies copied before source for caching; UID 10001; CPU-only native image and smaller remote client.",
            ],
            [
                "Build context",
                "Allowlisted serving files; exclude secrets, dataset, weights, Git metadata and training tools; no pip cache.",
            ],
            [
                "Services",
                "web + db; web uses db:5432 and waits for database health. Health probes check readiness.",
            ],
            [
                "Persistence",
                "Existing external PostgreSQL volume; read-only champion/policy mount; read-only root with 128 MiB tmpfs.",
            ],
            [
                "Ports / secrets",
                "Web 127.0.0.1:5000 and DB admin 127.0.0.1:5433; ignored environment file; no public database port.",
            ],
        ],
        [87, 441],
        compact=True,
    )
    h("6.4 Size reduction and layers")
    p(
        "docker image ls reports native <b>1.72 GB</b> versus client <b>395 MB</b>, about 77% smaller. "
        "Moving inference out of the web image removes TensorFlow, trading offline use for a network "
        "dependency. docker history shows the largest install layer falling from 1.22 GB to 63.9 MB."
    )
    row(
        [
            (shot("docker-warmup"), "Recorded Docker warm-up and cleanup."),
            (shot("docker-stack"), "Healthy services and preserved mounts."),
        ],
        height=119,
    )
    h("6.5 Failure diagnosis and recovery")
    p(
        "In the isolated native lab, a missing model mount caused FileNotFoundError for policy.json. "
        "Adding the read-only bundle restored HTTP 200, loaded=true and healthy PostgreSQL (Figure 31). "
        "Logs, shell and lifecycle commands were checked; earlier database recreation preserved history."
    )
    figure(
        shot("docker-failure"),
        "Actual missing-policy log and corrected health response.",
        height=105,
    )

    # 13: concise CI evidence and reproduction/handoff.
    page("Task 6 - Automated checks")
    h("6.6 Failing-then-passing CI lab")
    p(
        "The private 23L-0907_A2_Ops-ci-lab repository runs Python 3.11 and 3.12. The first commit "
        "deliberately expects 2+2=5 and fails; correcting the expectation to 4 makes both versions pass "
        "(Figures 32-33). Each result belongs to a separate committed workflow run."
    )
    row(
        [
            (shot("ci-red"), "Expected failure on both Python versions."),
            (shot("ci-green"), "Corrected test passes on both versions."),
        ],
        height=141,
    )
    h("Project tests and Docker build checks")
    p(
        "Main CI checks repository hygiene, Ruff, synthetic tests, native/client Docker builds and an "
        "isolated Compose upload/persistence test. The graded implementation passed <b>141 tests</b> "
        "and Docker/Compose checks. The README includes its CI badge; the screenshot below shows the "
        "project workflow, and the exact graded-revision run is linked below."
    )
    figure(
        shot("ci-project"),
        "Passing main project workflow with build and test jobs.",
        height=144,
    )
    p(
        "<link href='https://github.com/Hanzala-Shadow/23L-0907_A2_Ops/actions/runs/40127571425'>"
        "Graded-revision CI run 40127571425</link>. Routine CI uses generated runner credentials. "
        "The separately authorized cloud API key is an encrypted GitHub Actions secret used only "
        "by the manual Postman check; it is never embedded in source or reports."
    )
    h("Reproduction and handoff")
    p(
        "Check free space with scripts/check_disk.py, restore missing assets using dvc pull, then "
        "use docker compose up -d --no-build --wait with the configured cloud override. Full setup "
        "is in README.md and docs/cloud-integration.md. The report builder reads saved results only. "
        "Do not rerun the one-time test evaluator to regenerate the report."
    )
    p(
        "For submission: review this PDF, grant the grader access to both private repositories and "
        "share the API key privately if required. Source pushes contain no credentials, dataset/model "
        "binaries, agent files or decorative icons.",
        "small",
    )

    # 14: deployment steps, live readiness and genuine external Postman run.
    page("Task 7 - Cloud deployment")
    p(
        "Vercel Hobby hosts the key-protected model API while Flask and PostgreSQL stay local. "
        "The float32 LiteRT export avoids shipping TensorFlow in the function and matched every "
        "validation decision. Render required card verification; Vercel was used without paid add-ons."
    )
    table(
        ["Step", "Deployment action"],
        [
            [
                1,
                "Lock champion and threshold; export float32 LiteRT and verify validation parity.",
            ],
            [
                2,
                "DVC-version the export; prepare a model-only archive and pin its SHA-256 manifest.",
            ],
            [
                3,
                "Create the Hobby Python project in fra1 with pinned lightweight dependencies.",
            ],
            [4, "Set cloud.wsgi:app; fetch and verify the model bundle during build."],
            [
                5,
                "Supply a short-lived model-only URL and a new API key as encrypted settings.",
            ],
            [
                6,
                "Deploy reviewed source; confirm READY and public health. POST /predict requires the key.",
            ],
            [
                7,
                "Run the Postman collection from a separate GitHub-hosted network; retain safe results.",
            ],
        ],
        [34, 494],
        compact=True,
    )
    row(
        [
            (shot("cloud-health"), "Public cloud readiness with model identity."),
            (shot("postman-output"), "Actual external Postman CLI result."),
        ],
        height=115,
    )
    h("External Postman validation")
    postman = read("reports/labs/postman/verification.json")
    table(
        ["Request", "HTTP", "Response ms"],
        [[r["name"], r["http_status"], r["response_ms"]] for r in postman["results"]],
        [354, 70, 104],
        compact=True,
    )
    p(
        "Postman CLI 1.76.0 ran six requests and <b>13 assertions with zero failures</b> on a hosted "
        "Ubuntu runner. Generated red, blue and white images test wiring, not accuracy. "
        "The versioned collection is postman/visionguard.postman_collection.json."
    )
    p(
        "Live URL: <link href='https://23l-0907-a2-ops-api.vercel.app/health'>"
        "https://23l-0907-a2-ops-api.vercel.app/health</link>. "
        "Cold starts and free-tier quotas affect availability; a new build needs a refreshed model-download URL. "
        "Automatic Git deployment is disabled.",
        "small",
    )

    # 15: complete integration, error handling, and brief references.
    page("Task 8 - Local-to-cloud integration")
    p(
        "After local validation, Flask sends the image over verified HTTPS to Vercel. The client uses "
        "a <b>3-second connect timeout and 45-second read timeout</b>, no redirects and no automatic "
        "POST retries. It bounds JSON to 16 KiB and checks version, hashes, saved threshold, score, "
        "verdict, confidence and dimensions before recording success."
    )
    table(
        ["Generated input", "AI probability", "Verdict", "History"],
        [
            ["Red", "0.0782903", "real", "saved success"],
            ["Blue", "0.1078499", "real", "saved success"],
            ["White", "0.0190690", "real", "saved success"],
        ],
        [135, 135, 100, 158],
        compact=True,
    )
    row(
        [
            (shot("app-red"), "Red input result."),
            (shot("app-blue"), "Blue input result."),
            (shot("app-result"), "White input result."),
        ],
        height=94,
    )
    p(
        "All three generated inputs matched the local LiteRT reference within 0.0001. The browser "
        "upload/result/history flow was checked; no dataset image was sent for these integration tests."
    )
    table(
        ["Condition", "HTTP", "Outcome"],
        [
            [
                "Wrong / corrupt / oversized file",
                "415 / 400 / 413",
                "Friendly saved rejection",
            ],
            ["Connect or read timeout", "504", "cloud_timeout; no verdict"],
            [
                "Network, authentication or busy service",
                "503",
                "Safe error and retry guidance",
            ],
            ["Invalid or mismatched response", "502", "Rejected score; no success row"],
        ],
        [225, 95, 208],
        compact=True,
    )
    row(
        [
            (shot("app-wrong-type"), "Wrong type rejected."),
            (shot("app-corrupt"), "Corrupt image rejected."),
            (shot("app-too-large"), "Oversized image rejected."),
        ],
        height=87,
    )
    row(
        [
            (shot("integration-results"), "Recorded cloud results and database IDs."),
            (shot("app-timeout"), "Saved timeout with a readable history link."),
        ],
        height=105,
    )
    p(
        "Injected timeout/network failures wrote error rows without interrupting production services. "
        "Original PostgreSQL rows and native model files remained unchanged. Uploads are not saved, "
        "logs rotate and disk guards preserve the <b>2 GiB reserve</b>.",
        "small",
    )
    p(
        "<b>References:</b> "
        "<link href='https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images'>Kaggle dataset</link>; "
        "<link href='https://keras.io/guides/transfer_learning/'>Keras</link>; "
        "<link href='https://mlflow.org/docs/latest/index.html'>MLflow</link>; "
        "<link href='https://doc.dvc.org/'>DVC</link>; "
        "<link href='https://docs.docker.com/compose/'>Docker Compose</link>; "
        "<link href='https://docs.github.com/en/actions'>GitHub Actions</link>; "
        "<link href='https://vercel.com/docs'>Vercel</link>; "
        "<link href='https://learning.postman.com/docs/postman-cli/postman-cli-overview/'>Postman CLI</link>. "
        "Full evidence mapping: docs/assignment-checklist.md.",
        "small",
    )

    output = ROOT / "output/pdf/23L-0907_VisionGuard.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#bbcbd1"))
        canvas.line(42, 37, 570, 37)
        canvas.setFont("Body", 7)
        canvas.setFillColor(colors.HexColor("#50616a"))
        canvas.drawString(42, 24, "Syed Muhammad Hanzala | 23L-0907_A2_Ops")
        canvas.drawRightString(570, 24, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output),
        pagesize=(612, 792),
        leftMargin=42,
        rightMargin=42,
        topMargin=38,
        bottomMargin=50,
        title="23L-0907 VisionGuard assignment report",
        author="Syed Muhammad Hanzala",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    assert output.stat().st_size < 10 * 1024**2, (
        "Keep the report below the Git file-size limit"
    )
    assert shutil.disk_usage(ROOT).free >= 2 * 1024**3
    print(
        json.dumps(
            {
                "pdf": str(output),
                "source_commit": commit,
                "figures": figures,
                "bytes": output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
