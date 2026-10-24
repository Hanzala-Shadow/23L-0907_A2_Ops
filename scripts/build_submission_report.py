"""Build the assignment PDF from saved results and genuine UI evidence."""

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

from PIL import Image as PILImage
from PIL import ImageChops
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    if shutil.disk_usage(ROOT).free < 2.125 * 1024**3:
        raise RuntimeError("Keep the 2 GiB reserve plus report headroom")
    for name, path in [
        ("Body", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("BodyBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, path))
    styles = getSampleStyleSheet()
    for name in ["Normal", "BodyText", "Title", "Heading1", "Heading2", "Heading3"]:
        styles[name].fontName = "BodyBold" if name.startswith("Heading") else "Body"
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 8
    styles["BodyText"].allowWidows = 0
    styles["BodyText"].allowOrphans = 0
    styles["Heading1"].fontSize = 21
    styles["Heading1"].leading = 26
    styles["Heading1"].textColor = colors.HexColor("#16394c")
    styles["Heading2"].fontSize = 13
    styles["Heading2"].leading = 18
    styles["Heading2"].spaceBefore = 15
    styles.add(
        ParagraphStyle(
            "Caption",
            fontName="Body",
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#43545d"),
            spaceAfter=12,
        )
    )
    styles.add(ParagraphStyle("Cell", fontName="Body", fontSize=8, leading=10.5))
    styles.add(
        ParagraphStyle(
            "Reference", fontName="Body", fontSize=7.6, leading=10, spaceAfter=5
        )
    )
    styles.add(
        ParagraphStyle(
            "CodeBlock",
            fontName="Mono",
            fontSize=7.4,
            leading=10,
            backColor=colors.HexColor("#f0f3f5"),
            borderPadding=7,
            spaceAfter=10,
            splitLongWords=True,
        )
    )
    story = []
    figure_number = 0
    index = []

    def p(value, style="BodyText"):
        story.append(Paragraph(value, styles[style]))

    def h(value):
        p(value, "Heading2")

    def chapter(value):
        if story:
            story.append(PageBreak())
        p(value, "Heading1")
        index.append(value)

    def code(value):
        p(escape(value).replace("\n", "<br/>"), "CodeBlock")

    def table(headers, rows, widths=None):
        data = [
            [Paragraph(escape(str(x)), styles["Cell"]) for x in row]
            for row in [headers, *rows]
        ]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
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
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#90a5ad")),
                ]
            )
        )
        if len(data) <= 8:
            story.append(KeepTogether([t, Spacer(1, 10)]))
        else:
            story.extend([t, Spacer(1, 10)])

    def figure(path, caption, max_height=300):
        nonlocal figure_number
        source = ROOT / path
        if not source.is_file():
            raise FileNotFoundError(source)
        figure_number += 1
        for i in range(len(story) - 1, -1, -1):
            previous = story[i]
            if isinstance(previous, Paragraph) and previous.style.name == "BodyText":
                original = getattr(previous, "report_text", previous.text)
                references = [*getattr(previous, "report_figures", []), figure_number]
                label = "Figure" if len(references) == 1 else "Figures"
                updated = Paragraph(
                    original + f" <i>({label} {', '.join(map(str, references))}.)</i>",
                    previous.style,
                )
                updated.report_text, updated.report_figures = original, references
                story[i] = updated
                break
            if isinstance(previous, Paragraph) and previous.style.name == "Heading1":
                break
        im = PILImage.open(source).convert("RGB")
        if source.stem in {"app-health", "cloud-health"}:
            # Keep the full JSON and viewer header; remove empty white space and
            # the scrollbar from the report asset. Original captures are retained.
            im = im.crop((0, 0, im.width - 16, im.height))
            bounds = ImageChops.difference(
                im, PILImage.new("RGB", im.size, "white")
            ).getbbox()
            if bounds:
                im = im.crop((0, 0, im.width, min(im.height, bounds[3] + 15)))
        w, ht = im.size
        scale = min(WIDTH / w, max_height / ht)
        assets = ROOT / "reports/training-runtime/submission/report-assets"
        assets.mkdir(parents=True, exist_ok=True)
        prepared = assets / (str(figure_number) + ".png")
        im.thumbnail((round(w * scale * 2.5), round(ht * scale * 2.5)))
        if (
            "/data/" in path
            or "misclassified_examples.png" in path
            or "augmented_batch.png" in path
            or "/submission/evidence/" in path
        ):
            prepared = prepared.with_suffix(".jpg")
            im.save(prepared, quality=92, subsampling=0, optimize=True)
        else:
            im.save(prepared, optimize=True)
        im.close()
        picture = Image(str(prepared), width=w * scale, height=ht * scale)
        story.append(
            KeepTogether(
                [
                    picture,
                    Spacer(1, 5),
                    Paragraph(
                        f"Figure {figure_number}. {escape(caption)}", styles["Caption"]
                    ),
                ]
            )
        )

    def shot(name, caption, max_height=310):
        figure("reports/submission/evidence/" + name + ".png", caption, max_height)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    project = "23L-0907_A2_Ops"
    now = datetime.now(ZoneInfo("Asia/Karachi"))
    chapter("VisionGuard")
    p("AI-generated and real-labeled image classification", "Heading2")
    story.append(Spacer(1, 15))
    table(
        ["Submission", "Value"],
        [
            ["Name", "Syed Muhammad Hanzala"],
            ["Roll number / project", "23L-0907 / " + project],
            ["Prepared", now.strftime("%d %B %Y, %H:%M Asia/Karachi")],
            ["Graded source revision", commit],
            ["GitHub", "https://github.com/Hanzala-Shadow/23L-0907_A2_Ops"],
            ["DagsHub", "https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops"],
            ["Live model API", "https://23l-0907-a2-ops-api.vercel.app/predict"],
            ["Public readiness", "https://23l-0907-a2-ops-api.vercel.app/health"],
            ["Local Flask UI", "http://127.0.0.1:5000/"],
        ],
        [135, 393],
    )
    p(
        "The source revision above identifies the code and evidence used for this report. "
        "The PDF itself is added in a following documentation-only commit, avoiding a self-referential commit hash."
    )
    h("Assistance disclosure")
    p(
        "OpenAI Codex assisted with environment setup, implementation, debugging, tests, experiment execution, "
        "deployment, and drafting this report. Official TensorFlow/Keras, MLflow, DVC, Docker, GitHub, Vercel "
        "and Postman documentation informed the tool configuration. The saved scripts, run receipts and logs "
        "provide the evidence behind the results. No claim is made that this work was completed without assistance."
    )
    p(
        "Both repositories remain private. The student will share grading access separately. The prediction "
        "endpoint requires its server-side API key; no credential is printed in this report."
    )

    chapter("Project outcome and reading guide")
    p(
        "The system audits and versions the Kaggle data, trains a custom CNN and MobileNetV2, compares eleven "
        "candidates, locks a validation-selected threshold, and evaluates two locked models once on test. "
        "The champion is MLflow registry version 2. A local Flask interface calls its LiteRT export on Vercel, "
        "then writes successful and failed attempts to the existing PostgreSQL database."
    )
    table(
        ["Task", "Evidence in this report"],
        [
            [
                "1",
                "Environment, inventory, EDA, duplicate policy, deterministic splits, DVC toy restore",
            ],
            [
                "2",
                "Input pipeline, CNN layers, two transfer phases, discipline, test results, ten errors, tuner",
            ],
            [
                "3",
                "Local and remote toy runs, manual/autolog tracking, artifacts, comparison, registry",
            ],
            [
                "4-5",
                "Inference lifetime, upload validation, UI, configuration, PostgreSQL trail",
            ],
            [
                "6",
                "Warm-up, image design, Compose, size optimization, recovery, red/green CI",
            ],
            [
                "7-8",
                "Cloud deployment steps, Postman from another network, local API integration",
            ],
        ],
        [65, 463],
    )
    p(
        "Task 7's direct instruction and Task 8 require localhost Flask calling a cloud model API; this is the "
        "implemented arrangement. The example pathway table describes other layouts too. Vercel's DNS-safe "
        "resource name is 23l-0907-a2-ops-api; the GitHub, DagsHub, MLflow and Compose project identity is "
        "23L-0907_A2_Ops. The Docker and DVC toy exercises were completed during the final audit rather than "
        "before the main implementation; their timestamps are retained."
    )
    p(
        "Optional Grad-CAM bonus work was not attempted. No training, threshold selection, conversion or "
        "deployment check reuses the held-out test set after the recorded final evaluation. Screenshots are "
        "genuine browser captures; transcript screenshots are explicitly labeled recorded command output."
    )

    chapter("Task 1 - Data understanding and versioning")
    h("1.1 Isolated CPU environment")
    p(
        "The Python virtual environment is .venv. TensorFlow CPU 2.19.1, Keras 3.15.1, MLflow 2.22.5 and "
        "DVC 3.67.1 are pinned with the other dependencies. GPU discovery returned an empty list. CPU thread "
        "counts are bounded for this machine. The environment smoke check trained synthetic tensors and "
        "verified Keras/MLflow save-load, Flask and SQLAlchemy before downloading the assignment data."
    )
    shot(
        "environment",
        "Recorded environment command output with project, user, time, versions and GPU result.",
    )
    code(
        "python3 -m venv .venv\n.venv/bin/python -m pip install -r requirements/locked.txt\n.venv/bin/python scripts/check_environment.py --smoke\n.venv/bin/python scripts/check_disk.py --minimum-free-gib 2"
    )
    p(
        "venv isolates packages; the lock records exact versions; the smoke flag exercises functionality; "
        "the disk check stops unsafe work. With the environment, raw images and DVC cache retained, the "
        "reserve is checked before training, artifact work and uploads. No large local rebuild was needed for submission."
    )
    h("1.2 Audit and reproducible splits")
    audit = read("reports/data/audit_summary.json")
    rows = []
    for label, a in audit["classes"].items():
        rows.append(
            [
                label,
                a["total_files"],
                a["unreadable_files"],
                " / ".join(str(a["width"][k]) for k in ["min", "median", "max"]),
                " / ".join(str(a["height"][k]) for k in ["min", "median", "max"]),
                " / ".join(
                    str(a["file_size_bytes"][k]) for k in ["min", "median", "max"]
                ),
            ]
        )
    table(
        [
            "Folder label",
            "Files",
            "Corrupt",
            "Width min/med/max",
            "Height min/med/max",
            "Bytes min/med/max",
        ],
        rows,
        [85, 35, 44, 110, 110, 144],
    )
    p(
        "The custom walker records class, bytes, dimensions, colour mode, decoded format and SHA-256 for "
        "every file in reports/data/image_manifest.csv. All 975 files decode. One exact duplicate pair has the "
        "same class: image33.jpg and image34.jpg. The raw source is preserved, while split preparation excludes "
        "image34.jpg. There are 974 unique hashes and no conflicting exact-hash labels."
    )
    table(
        ["Split", "AI-generated", "Real-labeled", "Total"],
        [["Train", 376, 305, 681], ["Validation", 81, 65, 146], ["Test", 81, 66, 147]],
        [135, 130, 130, 133],
    )
    p(
        "A SHA-256 ranking of seed 42 and each content hash makes the stratified 70/15/15 lists independent "
        "of input row order. Largest-remainder allocation determines integer counts. Checks reject overlapping "
        "paths or hashes and verify source bytes. A forced DVC reproduction produced identical lists. Exact "
        "hashes do not detect differently encoded near-duplicates or establish true image provenance."
    )
    h("1.3 Visual EDA")
    p(
        "The grid shows the collection includes photographs, paintings, composites and web graphics. Folder "
        "labels are the task target, not independently verified origin. This visual diversity is a likely source "
        "of errors and limits claims about real-world provenance."
    )
    figure(
        "reports/data/sample_grid.png",
        "Seeded samples per folder class; source labels are not verified provenance.",
        390,
    )
    for field, explanation in [
        (
            "width",
            "Both classes have varied widths, with some very large files. AI-labeled widths cluster near common generation sizes, while real-labeled widths are also heterogeneous. Resizing is necessary for bounded CPU cost.",
        ),
        (
            "height",
            "Image heights vary strongly and include portrait and landscape examples. A square training tensor would distort subjects if stretched. White letterboxing preserves the aspect ratio while fixing tensor dimensions.",
        ),
        (
            "file_size_bytes",
            "File sizes have a long tail and depend on compression and content as well as dimensions. A small compressed file can still decode to many pixels. Serving therefore checks both byte count and pixel count.",
        ),
    ]:
        p(explanation)
        figure(
            f"reports/data/{field}_histogram.png",
            f"Per-class {field.replace('_', ' ')} distributions from the complete audit.",
            240,
        )
    h("1.4 DVC toy lab and project storage")
    dvc = read("reports/labs/dvc/verification.json")
    p(
        "The separate tiny CSV lab initialized Git and DVC in an empty folder, committed a two-row CSV pointer, "
        "added a third row, and committed the second pointer. Both versions were pushed on the dvc-lab branch "
        "of the existing private project. A local folder outside that toy repository is its remote. Returning to "
        "the first commit restored the older CSV. Deleting both its workspace file and its toy cache, then "
        "running dvc pull, restored exactly the old bytes. Production images and cache were untouched."
    )
    table(
        ["Version", "Git commit"],
        [
            ["Initial", dvc["version_commits"][0]],
            ["Updated", dvc["version_commits"][1]],
        ],
        [90, 438],
    )
    code(dvc["pointer"])
    p(
        "The pointer stores a checksum, byte size and relative output path. Git versions this small text; "
        "DVC stores large content separately by checksum. The real project uses DagsHub's S3-compatible DVC "
        "store. A previous empty-cache restore verified all 975 source hashes and five split files. Models "
        "are also DVC-versioned. Credentials stay in ignored .dvc/config.local, never shared configuration."
    )
    shot(
        "dvc-lab",
        "Recorded DVC two-version history, older-version checkout and clean-cache pull verification.",
    )

    chapter("Task 2 - Model development")
    h("2.1 Input pipeline and augmentation")
    p(
        "Split CSVs supply paths, labels and original hashes. The shared Pillow decoder applies EXIF "
        "orientation, converts to RGB, composites transparency on white, then bilinearly resizes into a "
        "centered 128 x 128 letterbox. tf.numpy_function bridges that decoder into tf.data; shape is fixed "
        "to 128 x 128 x 3. Decoded uint8 tensors are cached in RAM, training is shuffled with seed 42, "
        "batches contain 16 images and one batch is prefetched. Decoding uses one worker and the two caches "
        "use roughly 39 MiB. No resized image dataset is written to disk."
    )
    p(
        "128 pixels bounds CPU time but loses fine artifacts. Aspect ratio is preserved through padding. "
        "Training-only Keras layers flip horizontally, rotate up to 18 degrees and zoom up to 10%. They are "
        "inactive for validation and inference. Normalization is saved inside each model."
    )
    figure(
        "reports/training/cnn_baseline/augmented_batch.png",
        "Original and augmented training images; no validation or test augmentation.",
        330,
    )
    h("2.2 Custom CNN baseline")
    p(
        "Three convolution blocks have 16, 32 and 64 filters. A convolution learns local spatial patterns; "
        "BatchNorm stabilizes activations; ReLU adds nonlinearity; max pooling reduces spatial cost. Biases "
        "are omitted before BatchNorm. Global average pooling replaces an expensive flatten layer. A "
        "64-unit head and dropout 0.3 reduce capacity; one sigmoid estimates the AI-generated probability. "
        "The model has 28,145 total parameters. Binary cross-entropy matches the target; Adam starts at "
        "0.001. Accuracy, precision, recall and ROC-AUC make different errors visible."
    )
    with (ROOT / "reports/training/cnn_baseline/layers.csv").open() as f:
        layers = list(csv.DictReader(f))
    table(
        ["Layer", "Type", "Output shape", "Parameters"],
        [
            [r[k] for k in ["layer", "type", "output_shape", "parameters"]]
            for r in layers
        ],
        [128, 137, 193, 70],
    )
    shot(
        "model-summary",
        "Recorded Keras summary for the custom CNN, including parameter totals.",
    )
    h("2.3 Frozen backbone and fine-tuning")
    p(
        "ImageNet MobileNetV2 is loaded without its original classifier. Its head uses global average "
        "pooling, Dense(64), dropout 0.3 and a sigmoid. Its expected normalization is pixel / 127.5 - 1, "
        "saved as a Rescaling layer; the CNN instead uses pixel / 255. Passing already normalized pixels "
        "to these models would apply normalization twice and break the training-serving contract."
    )
    p(
        "The frozen stage trains only 82,049 head parameters at learning rate 0.001. The second stage "
        "starts from the best frozen checkpoint and recompiles Adam at 0.00001. Only block_16_expand, "
        "block_16_depthwise and block_16_project become trainable: 551,489 trainable parameters including "
        "the head. Earlier blocks, Conv_1 and every BatchNorm layer remain frozen. Calling the backbone "
        "with training=False preserves moving statistics while selected convolution weights receive gradients. "
        "Weight hashes verified those boundaries. Fine-tuning worsened validation loss, so it was not assumed superior."
    )
    figure(
        "reports/training/mobilenetv2_frozen/training_curves.png",
        "Frozen-head loss and accuracy; best validation-loss epoch is marked.",
        235,
    )
    figure(
        "reports/training/mobilenetv2_finetuned/training_curves.png",
        "Last-block fine-tuning curves; earlier weights and BatchNorm remained frozen.",
        235,
    )
    h("2.4 Training discipline")
    p(
        "Seed 42 fixes Python, NumPy and TensorFlow random choices. Deterministic TensorFlow operations, "
        "two intra-op threads, one inter-op thread, disabled XLA and disabled oneDNN reduce variation. Exact "
        "cross-hardware equivalence is not guaranteed. Every stage checkpoints minimum validation loss, "
        "stops after no improvement and reduces the learning rate on a plateau. Only the best checkpoint "
        "is used for validation predictions; the final epoch is not silently substituted."
    )
    table(
        ["Run", "Best epoch", "Epochs completed", "Best validation loss"],
        [
            [
                r["model"],
                r["best_epoch"],
                r["epochs_completed"],
                f"{r['validation_loss']:.5f}",
            ]
            for r in read("reports/training/model_comparison.json")["models"]
        ],
        [205, 80, 110, 133],
    )
    figure(
        "reports/training/cnn_baseline/training_curves.png",
        "CNN early stopping after six epochs; the best checkpoint was epoch 1.",
        235,
    )
    h("2.5 Locked threshold and one test evaluation")
    p(
        "The champion was selected from eleven candidates by minimum validation cross-entropy. Its "
        "threshold 0.6196606457233429 minimizes 2 x false positives + false negatives on validation. A "
        "false accusation that a real-labeled image is fake is assigned twice the cost of missing an AI "
        "image. This is a stated product assumption, not an empirically measured harm ratio. Model and "
        "threshold were locked before test evaluation; the CNN received its own validation-selected threshold."
    )
    results = []
    for role in ["cnn_baseline", "champion"]:
        m = read(f"reports/evaluation/{role}/metrics.json")
        results.append(
            [
                role,
                *[
                    f"{m[k]:.4f}"
                    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                ],
            ]
        )
    table(
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        results,
        [153, 75, 75, 75, 75, 75],
    )
    p(
        "Both locked models received exactly one inference pass on 147 test images (66 real-labeled, "
        "81 AI-generated). Subsequent figures use the saved probability arrays. Identity receipts bind "
        "the model hash, split manifest, threshold and selection-plan hash; a durable ledger refuses "
        "another test pass. The champion confusion matrix is TN=51, FP=15, FN=26, TP=55. The CNN is weak "
        "(TN=56, FP=10, FN=78, TP=3), and that result is retained rather than retuned on test."
    )
    for role in ["cnn_baseline", "champion"]:
        figure(
            f"reports/evaluation/{role}/confusion_matrix.png",
            f"{role}: confusion matrix from the single saved test pass.",
            235,
        )
        figure(
            f"reports/evaluation/{role}/roc.png",
            f"{role}: test ROC using the same saved probabilities.",
            235,
        )
    p(
        "Champion accuracy dropped from 80.82% on validation to 72.11% on test. The data are small and "
        "varied, and validation was repeatedly reused during search. Neither these metrics nor a confident "
        "single prediction establish image origin. The observed test result did not change the locked model."
    )
    h("2.6 Ten misclassified examples")
    figure(
        "reports/evaluation/misclassified_examples.png",
        "The first ten errors in fixed test-manifest order, rendered from the saved evaluation selection.",
        400,
    )
    p(
        "All ten examples are false negatives because AI-labeled paths occur first in the fixed manifest; "
        "this is not a representative sample of all error types. The full evaluation also contains 15 false positives."
    )
    hypotheses = [
        "A multi-photo web screenshot may favor photo and page-layout cues.",
        "A nebula-like painterly texture overlaps with real-labeled artwork.",
        "Photographic context around framed artwork may dominate its small generated region.",
        "Fur and garden textures look familiar; probability 0.617 is just below the 0.620 threshold.",
        "A product graphic has a large background and small artworks; resizing suppresses detail.",
        "An infographic mixes text and cutouts, unlike a single isolated generated image.",
        "A grainy monochrome portrait may evoke real photographic cues.",
        "A saturated landscape has ordinary scene structure shared with real-labeled artwork.",
        "A plausible mountain scene may favor semantic content over synthesis artifacts.",
        "A promotional banner combines text and a small portrait; its 0.548 score misses the cutoff.",
    ]
    table(
        ["Example", "Visual hypothesis"],
        [[i + 1, t] for i, t in enumerate(hypotheses)],
        [65, 463],
    )
    p(
        "These observations are hypotheses, not causal attribution results. They motivate a future independent "
        "data review and are not used to alter the current test evaluation."
    )
    h("2.7 Keras Tuner tool lab and eight project trials")
    toy = read("reports/training/toy_tuner/summary.json")
    p(
        "The built-in MNIST toy lab uses 6,000 training and 1,000 validation examples drawn only from MNIST's "
        "training partition. RandomSearch tests two learning rates and two hidden sizes for three epochs. "
        "The best setting is learning rate 0.01 and 64 units, validation loss 0.24696. Its four runs are logged separately on DagsHub."
    )
    table(
        ["Toy trial", "Learning rate", "Units", "Validation loss", "Accuracy"],
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
        [70, 110, 70, 145, 133],
    )
    trials = []
    for i in range(8):
        hp = read(f"reports/training/keras_tuner/trial_{i}/configuration.json")[
            "hyperparameters"
        ]
        m = read(f"reports/training/keras_tuner/trial_{i}/metrics.json")
        trials.append(
            [
                i,
                hp["learning_rate"],
                hp["dropout"],
                hp["unfrozen_layers"],
                f"{m['validation_loss']:.5f}",
                m["best_epoch"],
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
        [45, 105, 80, 115, 100, 83],
    )
    p(
        "The project search covers the complete two-level factorial grid: rates 0.00001/0.0001, dropout "
        "0.2/0.5 and one/three unfrozen final-block convolution layers. All trials start from the same frozen "
        "checkpoint. Trial 7 wins at 0.0001, dropout 0.5 and three unfrozen convolutions, validation loss "
        "0.485135. Dropout has the largest marginal effect: mean validation loss falls by 0.02322 at 0.5. "
        "Learning rate and unfrozen count have smaller average effects, but interactions and a small sample "
        "limit causal claims. Selecting repeatedly on validation can overfit that set, even with early stopping."
    )
    p(
        "Evidence: docs/tuning.md and reports/training/keras_tuner/summary.json. No search used test images."
    )

    chapter("Task 3 - MLflow and DagsHub")
    h("3.1 Local MLflow tool lab")
    p(
        "A 16-unit Keras network trained on 1,200 MNIST training examples with 200 validation examples for "
        "two epochs. Two local runs differ only in Adam rate: 0.001 versus 0.01. Each logs parameters, epoch "
        "metrics, a loss plot and the model. The local SQLite-backed MLflow UI shows both runs and their "
        "comparison. At this tiny budget, validation accuracy is 0.54 versus 0.80, respectively; this toy result "
        "is not comparable with VisionGuard."
    )
    shot("mlflow-comparison", "Local MLflow UI comparing the two real MNIST runs.")
    shot(
        "mlflow-comparison-metrics",
        "Local comparison view with run times, learning rates and epoch-final metrics.",
    )
    table(
        ["Term", "Meaning in this project"],
        [
            ["Experiment", "A named workspace grouping related runs."],
            ["Run", "One recorded training attempt with an identity and lifecycle."],
            [
                "Parameter",
                "A configuration value, such as learning rate, fixed for the attempt.",
            ],
            ["Metric", "A measured numeric result, often recorded at each epoch."],
            ["Artifact", "A saved file such as a model, plot, CSV or package list."],
            [
                "Tag",
                "Searchable context, such as source commit, seed or logging method.",
            ],
        ],
        [95, 433],
    )
    h("3.2 DagsHub connection and toy repeat")
    p(
        "The private DagsHub project mirrors GitHub. Its MLflow tracking URI is "
        "https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops.mlflow. The client reads credentials from "
        "environment variables populated by ignored .env. git check-ignore confirms .env, .env.cloud and "
        ".dvc/config.local stay untracked. A 0.001-rate toy run was repeated on DagsHub, with two epoch "
        "records, its plot and model independently verified through the tracking API."
    )
    shot(
        "dagshub-data",
        "Private DagsHub project connected to GitHub with versioned data pointers.",
    )
    shot(
        "dagshub-toy",
        "DagsHub toy run, separate from VisionGuard training experiments.",
    )
    p(
        "A first remote attempt accidentally retained the local tracking address: MLflow set_tracking_uri "
        "updates the process environment. The correction captures the cloud address before selecting local "
        "storage. The local diagnostic run is labeled, and only the corrected remote run is claimed as DagsHub evidence."
    )
    h("3.3 Manual logging and TensorFlow autologging")
    p(
        "The CNN uses a Keras callback to log epoch loss, accuracy, AUC, precision, recall and learning rate. "
        "It explicitly adds seed, source commit, DVC checksum, split artifacts and selected checkpoint. The "
        "two MobileNet stages use TensorFlow autologging for fit parameters, optimizer data, epoch metrics "
        "and TensorBoard evidence. Automatic final-model upload is disabled; the best validation checkpoint "
        "is logged explicitly. Autologging does not replace project-specific provenance, layer-boundary "
        "checks, cost-based threshold choice or exact validation/test metrics. Tuner trials use explicit logging."
    )
    shot(
        "dagshub-manual-run",
        "CNN run evidence: manually logged metrics and project context.",
    )
    shot(
        "dagshub-autolog-run",
        "MobileNet frozen-stage evidence with framework autologging and explicit provenance.",
    )
    h("3.4 Required artifacts")
    p(
        "Training runs include the native Keras checkpoint, MLflow model, configuration, package list, split "
        "lists, layer summary, curves and validation outputs. The final evaluation run stores both models' "
        "test confusion matrices, ROC curves and saved predictions, and final test metrics are attached to "
        "the original training runs. Keeping evaluation separate preserves the original training provenance."
    )
    shot(
        "dagshub-artifacts",
        "MLflow artifact browser showing saved model and reporting files.",
    )
    h("3.5 Meaningful experiment comparison")
    table(
        ["Candidate", "Validation loss", "Accuracy at 0.5", "ROC-AUC"],
        [
            [
                r["model"],
                f"{r['validation_loss']:.5f}",
                f"{r['validation_accuracy']:.4f}",
                f"{r['validation_roc_auc']:.4f}",
            ]
            for r in [
                *read("reports/training/model_comparison.json")["models"],
                *[
                    {
                        "model": "tuner_" + str(i),
                        **read(f"reports/training/keras_tuner/trial_{i}/metrics.json"),
                    }
                    for i in range(8)
                ],
            ]
        ],
        [230, 100, 105, 93],
    )
    p(
        "The table compares the CNN, frozen MobileNet, last-block fine-tune and eight controlled tuner "
        "combinations. Minimum validation loss selects the model before cost-based threshold selection. "
        "The frozen network outperformed the first fine-tune; tuned dropout improved the best candidate. "
        "Final test results are reported as a check on the locked choice, not as another selection round."
    )
    shot(
        "dagshub-comparison",
        "Six selected DagsHub tuning runs with meaningful parameter differences; all eight are tabulated above.",
    )
    h("3.6 Registry and alias-based inference")
    p(
        "Registry 23L-0907_A2_Ops contains version 1 (frozen MobileNet) and version 2 (tuned MobileNet). "
        "The champion alias points to version 2. scripts/predict_registry.py independently loads "
        "models:/23L-0907_A2_Ops@champion, applies the shared preprocessing, and classifies a validation "
        "image. Its saved probability 0.9492709637 and threshold 0.6196606457 reproduce the selected bundle. "
        "No test image is used for that demonstration."
    )
    shot(
        "dagshub-registry",
        "Two registered versions, with champion assigned to version 2.",
    )
    shot("registry-load", "Recorded separate alias-load prediction and model identity.")

    chapter("Task 4 - Flask application")
    h("4.1 Inference lifetime and matching preprocessing")
    p(
        "visionguard/serving.py loads the native Keras champion once when the worker starts, validates "
        "its hash/policy and warms it with synthetic pixels. The native implementation remains available "
        "in the retained image and read-only bundle. Current Flask serving uses one RemoteModelService "
        "per worker and calls the cloud export; the Vercel process loads its interpreter once per warm "
        "instance. A cold start or another instance loads its own copy. DagsHub is not queried on requests."
    )
    p(
        "Loading weights per request would waste CPU, increase latency and memory churn, and risk selecting "
        "an inconsistent version. Both native and cloud paths use RGB white letterboxing and normalization "
        "inside the model. The float32 export matched all 146 validation decisions; test was not used for conversion. "
        "The two original known validation-image checks are recorded in reports/application/verification.json."
    )
    h("4.2 Routes and input validation")
    p(
        "GET / displays upload; POST /predict returns a saved result; GET /result/&lt;id&gt; reads it; "
        "GET /history paginates database records; GET /health checks readiness. The validator checks the "
        "form token, one file, sanitized basename, extension, decoded format, file bytes, decoded pixels "
        "and still-image content. The remote contract allows JPEG, PNG and WebP up to 4 MiB and 4 million "
        "pixels. Images stay in memory and are discarded; only metadata and results are retained."
    )
    for name, caption in [
        ("app-wrong-type", "Wrong file type: friendly saved rejection (HTTP 415)."),
        (
            "app-corrupt",
            "PNG-like bytes that cannot decode: friendly saved rejection (HTTP 400).",
        ),
        ("app-too-large", "Oversized file: friendly saved rejection (HTTP 413)."),
    ]:
        shot(name, caption)
    h("4.3 Upload, result and history interface")
    p(
        "The HTML/CSS interface offers an upload preview, verdict and confidence, a visible model caveat, "
        "and a history menu that reads PostgreSQL. The upload page explains cloud transmission and data "
        "retention. The API key is never exposed to browser JavaScript. Unknown result IDs return a friendly 404."
    )
    shot(
        "app-upload",
        "Local upload interface with limits and cloud-transmission disclosure.",
    )
    shot(
        "app-result",
        "Saved cloud-backed result showing verdict, confidence, model version and caveat.",
    )
    h("4.4 Environment settings, debug and server")
    p(
        "Model/policy paths, saved threshold, database URL, session secret, disk reserve and remote settings "
        "come from environment variables. A supplied threshold must match the saved policy. Debug is "
        "disabled to avoid interactive tracebacks and development behavior. The app was checked under "
        "Flask's server during development; Compose uses Gunicorn with one worker and two threads. Health "
        "reports a remote backend without claiming local weights are loaded."
    )
    shot(
        "app-health",
        "Current health JSON: PostgreSQL, Vercel model, saved threshold, debug off and disk reserve.",
    )

    chapter("Task 5 - PostgreSQL database trail")
    p(
        "The predictions table records UUID, UTC time, sanitized filename, image SHA-256/bytes/dimensions, "
        "status, error code/message, label, AI probability, confidence, threshold, processing time, model name, "
        "version, run ID, artifact hash and preprocessing policy. SQLAlchemy uses short transactions and "
        "parameterized queries. A successful prediction is not shown as saved if the database write fails. "
        "Invalid uploads and remote failures are recorded with safe messages; failed attempts have no verdict."
    )
    code(
        "SELECT created_at, status, label, confidence, model_version\nFROM predictions ORDER BY created_at DESC LIMIT 10;"
    )
    p(
        "The history menu renders these rows rather than a browser-only list. UTC is labeled in the UI. "
        "A named Docker volume preserves history across container recreation. Saved snapshots compared "
        "all original row fields, not just counts. All 11 original records survived Docker and cloud integration; "
        "the eight explicitly named integration/browser checks brought the count to 19. This final audit "
        "does not delete those records or rewrite old model hashes."
    )
    shot(
        "app-history",
        "PostgreSQL-backed history, including successful checks and safely recorded failures.",
    )

    chapter("Task 6 - Docker and continuous integration")
    h("6.1 Docker warm-up")
    p(
        "A disposable official postgres:17-alpine container demonstrated run, loopback port mapping, a new "
        "named volume, logs, shell access, listing, stop, start and restart. A temporary image tag was "
        "created and removed, then only the lab container and lab volume were deleted. The original image "
        "was retained because the real database uses it. Production containers and their volume were not touched."
    )
    table(
        ["Term", "Meaning"],
        [
            [
                "Image",
                "An immutable template of filesystem content and runtime defaults.",
            ],
            [
                "Container",
                "A running or stopped instance with its own process and writable state.",
            ],
            [
                "Layer",
                "A cached filesystem change produced by an image build instruction.",
            ],
            ["Volume", "Docker-managed storage that outlives a container."],
            [
                "Port mapping",
                "Forwarding a selected host address/port to a container's port.",
            ],
        ],
        [100, 428],
    )
    shot("docker-warmup", "Recorded ready-made image lab commands and cleanup receipt.")
    h("6.2 Dockerfiles and secret handling")
    p(
        "Dockerfile builds the CPU-only native server; Dockerfile.client builds the current lightweight "
        "remote client. Both start from pinned Python slim images, install pinned dependencies before copying "
        "source, run as UID 10001 and include an HTTP health probe. Dependency-first ordering preserves the "
        "large install layer when only source changes. Deny-by-default ignore files allow only serving "
        "inputs. Dataset images, weights, Git metadata, .env files and training tools do not enter image layers."
    )
    p(
        "The native image deliberately uses tensorflow-cpu and removes unused development headers. The "
        "client needs Flask, PostgreSQL, Pillow and Requests only. It retains the saved policy mount but "
        "does not install a local model runtime. The actual image is imported from a verified private CI "
        "artifact to avoid a local rebuild. The downloaded archive was removed after import."
    )
    h("6.3 Two-service Compose stack")
    p(
        "compose.yaml defines web and db; compose.cloud.yaml overrides web for remote inference. The setup "
        "script preserves database and session secrets and sets COMPOSE_FILE so normal Compose commands "
        "select both files. web uses db:5432 on the private Compose network, waits for the database health "
        "check, and publishes only 127.0.0.1:5000. PostgreSQL's administration mapping is loopback-only "
        "127.0.0.1:5433; there is no public bind. Passwords come from ignored .env and placeholders are "
        "provided in .env.example. The external named history volume is reused, and the champion directory "
        "is mounted read-only. The root filesystem is read-only with a bounded temporary memory mount."
    )
    shot(
        "docker-stack",
        "Recorded healthy services, UID, volume, read-only mount and loopback-only port bindings.",
    )
    h("6.4 Size and layer analysis")
    p(
        "The same docker image ls measurement reports 1.72 GB for the retained native TensorFlow image "
        "and 395 MB for the remote-client image, a reduction of about 77%. This optimization removes "
        "TensorFlow and numerical model dependencies from the local web image after moving inference to "
        "Vercel. It trades offline inference for network dependency; native serving remains available. "
        "Registry/archive compressed byte sizes are different measurements and are not mixed with these values."
    )
    p(
        "docker history identifies dependency installation as the largest layer in both images, with the "
        "native TensorFlow install layer at 1.22 GB versus the client's 63.9 MB install layer. Slim base, no pip cache, no compiler and an allowlisted build "
        "context reduce size further. The exact history output is in reports/labs/docker/verification.json."
    )
    h("6.5 Operations, failure diagnosis and recovery")
    p(
        "The isolated lab also ran the native web image against its disposable database. Omitting the "
        "champion mount made the Gunicorn worker exit. Logs showed FileNotFoundError for policy.json, "
        "which identified a missing serving bundle rather than a database problem. Adding the existing "
        "bundle as a read-only mount restored HTTP 200 with loaded=true and healthy PostgreSQL. Shell, "
        "logs, stop, start and restart were exercised for both services, and all temporary resources were removed."
    )
    shot(
        "docker-failure",
        "Actual missing-policy worker log and the corrected healthy native model response.",
    )
    p(
        "Earlier production checks also recreated PostgreSQL and confirmed the original rows after recovery. "
        "A temporary outage returned 503 and then recovered without restarting the Gunicorn worker. "
        "These receipts are retained under reports/container and reports/application."
    )
    h("6.6 GitHub Actions red/green lab and project CI")
    p(
        "The separate authorized private CI lab has a Python 3.11/3.12 matrix. Its initial arithmetic test "
        "expects 2+2=5 and fails on both versions; the next commit corrects the expectation to 4 and passes. "
        "The main repository workflow installs pinned dependencies, checks repository content, runs Ruff "
        "and synthetic tests, builds the native image, starts an isolated Compose stack, validates five "
        "uploads, recreates containers and checks persistence. It also builds the lightweight client and "
        "checks that TensorFlow, MLflow and DVC are absent. The README contains the CI badge."
    )
    for name, caption in [
        ("ci-red", "The toy matrix fails on Python 3.11 and 3.12 as intended."),
        ("ci-green", "The corrected toy test passes on both Python versions."),
        (
            "ci-project",
            "Main project CI passes lint, tests, Docker builds and Compose checks.",
        ),
    ]:
        shot(name, caption)
    p(
        "CI's generated database/session settings stay in ignored runner files. Actions secrets belong in "
        "GitHub repository or protected-environment secret storage and are referenced by name, never "
        "embedded in YAML. The separately authorized API-only secret is used solely by the manually "
        "triggered public API check. Its reports omit request headers and bodies and its temporary secret "
        "file is removed. No training account credential is available to that job."
    )

    chapter("Task 7 - Cloud model API deployment")
    h("Deployment design and limits")
    p(
        "The model API is on Vercel Hobby, with no billing details or paid add-ons. A float32 LiteRT export "
        "avoids the TensorFlow function-size problem while preserving the Keras champion. The runtime "
        "measured roughly 179 MiB locally and fits the provider's Python bundle allowance. Request bodies "
        "have a provider limit of 4.5 MB, so the app caps files at 4 MiB plus multipart headroom. Cold starts "
        "can add delay. Hobby is for personal non-commercial use and quota exhaustion can pause service; "
        "this is a free-tier demonstration rather than an availability guarantee."
    )
    h("Deployment steps and rationale")
    table(
        ["Step", "Action and purpose"],
        [
            [
                "1",
                "Lock the validation-selected champion; export float32 LiteRT with inference behavior and verify all 146 validation decisions. Do not use test.",
            ],
            [
                "2",
                "DVC-version the export; create a model-only private release archive. Pin archive and file SHA-256 values in cloud/model-manifest.json.",
            ],
            [
                "3",
                "Create the Vercel Hobby Python project in fra1. Keep the course project identity; use a DNS-safe deployment name.",
            ],
            [
                "4",
                "Use pyproject.toml to point at cloud.wsgi:app with pinned lightweight dependencies. vercel.json fetches and verifies the model at build time.",
            ],
            [
                "5",
                "Supply a fresh short-lived model-only download URL and a new API key as encrypted production settings. Keep all account tokens and local history off the provider.",
            ],
            [
                "6",
                "Deploy the reviewed source after CI. Keep automatic Git deployment disabled. Confirm READY, public health and authenticated prediction.",
            ],
            [
                "7",
                "Run the versioned Postman collection from GitHub's hosted runner, a network separate from the laptop, and retain the credential-free results.",
            ],
        ],
        [45, 483],
    )
    p(
        "The earlier Render Free route required card verification. No billing details were added and no "
        "Render service was created. Vercel was selected with the user's approval. Initial Vercel SSO "
        "protection was explicitly disabled only for this API project; POST /predict still requires its API key. "
        "The model download URL is refreshed before a future build because it expires. The serving API "
        "uses its bundled model and does not fetch DagsHub or the release on each request."
    )
    shot(
        "cloud-health",
        "Public cloud health response identifies the exact export, source model, version and threshold.",
    )
    h("Postman test and logged result from another network")
    postman = read("reports/labs/postman/verification.json")
    table(
        ["Request", "HTTP status", "Response ms"],
        [[r["name"], r["http_status"], r["response_ms"]] for r in postman["results"]],
        [330, 95, 103],
    )
    p(
        "Official Postman CLI 1.76.0 ran the local collection on a GitHub-hosted Ubuntu runner: six requests, "
        "13 assertions, zero failures. The three generated images returned consistent scores/verdicts and "
        "the pinned champion hash and threshold; missing authorization returned 401 and invalid text "
        "returned 415. No Postman account or cloud workspace was used. The key traveled only to the "
        "approved encrypted GitHub Actions secret and the intended Vercel prediction endpoint."
    )
    shot(
        "postman",
        "Successful Postman workflow from a separate network; the private artifact contains header/body-free results.",
    )
    shot(
        "postman-output",
        "Actual Postman CLI output: six requests and thirteen passing assertions.",
    )
    p(
        "Run: <link href='"
        + postman["workflow_run"]
        + "'>"
        + postman["workflow_run"]
        + "</link>. "
        "The importable collection is postman/visionguard.postman_collection.json. Add the API key to a "
        "private local Postman environment when testing manually; never paste it into the collection or URL."
    )

    chapter("Task 8 - Local Flask and cloud API integration")
    p(
        "Local validation runs first, then the server sends a generic upload filename over verified HTTPS "
        "to the pinned Vercel origin. It reuses connections per thread, uses a 3-second connect timeout and "
        "45-second read timeout, disables redirects and does not automatically retry POSTs. The client "
        "bounds JSON responses to 16 KiB and verifies model/version/hashes, saved threshold, finite "
        "probability, verdict, confidence and input dimensions before storing a success."
    )
    table(
        ["Generated input", "AI probability", "Verdict", "PostgreSQL result"],
        [
            ["Red", "0.0782903", "real", "saved success"],
            ["Blue", "0.1078499", "real", "saved success"],
            ["White", "0.0190690", "real", "saved success"],
        ],
        [115, 130, 105, 178],
    )
    p(
        "These solid-color images test wiring and parity, not model accuracy. Their cloud results match "
        "the local LiteRT reference within 0.0001. The same browser upload/result/history path was exercised "
        "with a generated white PNG. No dataset file was sent to the cloud for these checks."
    )
    shot(
        "integration-results",
        "Recorded multiple localhost-to-cloud results and their PostgreSQL identifiers.",
    )
    shot(
        "app-red",
        "Saved red-input cloud result, read back through the local Flask interface.",
        235,
    )
    shot(
        "app-blue",
        "Saved blue-input cloud result, read back through the same local interface.",
        235,
    )
    table(
        ["Failure", "HTTP result", "Recorded behavior"],
        [
            ["Connect/read timeout", "504", "cloud_timeout; no successful verdict"],
            [
                "Network/TLS/upstream failure",
                "503",
                "cloud_unavailable; safe retry guidance",
            ],
            ["Busy/rate-limited model", "503", "cloud_busy; Retry-After"],
            [
                "Wrong API key",
                "503",
                "cloud_authentication; connection needs attention",
            ],
            [
                "Malformed or mismatched result",
                "502",
                "cloud_invalid_response; no accepted score",
            ],
        ],
        [190, 75, 263],
    )
    p(
        "Timeout and network failures were injected into isolated Flask test applications sharing PostgreSQL; "
        "the running web container and cloud service were not interrupted. Both failures wrote error rows "
        "with null prediction fields, and history remained readable. Health probes have short bounded "
        "timeouts and cache readiness. There is no silent local-model fallback. All 11 original rows and "
        "the native champion files remained byte-for-byte unchanged."
    )
    shot(
        "app-timeout",
        "Saved timeout error is readable through the live local app and links back to history.",
    )
    p(
        "The production client uses the same local PostgreSQL volume, persistent session secret and saved "
        "threshold. Upload guards require 2 GiB plus a 128 MiB safety buffer. Logs rotate, uploaded files "
        "are not persisted, and disposable lab resources were removed. The minimum observed free space "
        "during the final Docker lab was 2.41 GiB, above the reserve."
    )

    chapter("Submission checklist and reproducibility")
    table(
        ["Item", "State"],
        [
            [
                "Named PDF",
                "23L-0907_VisionGuard.pdf, with name, links, source hash, assistance disclosure and task chapters",
            ],
            [
                "GitHub / DagsHub",
                "Private repositories with clean commits, DVC metadata, experiments and registry; student will grant grading access",
            ],
            [
                "Live API",
                "Public health; key-protected POST /predict tested from a separate GitHub-hosted network",
            ],
            [
                "Local UI/history",
                "Flask to Vercel integration active; original PostgreSQL rows and native model preserved",
            ],
            [
                "Evaluation integrity",
                "Two locked model test passes, one per model; no later test reuse",
            ],
            [
                "Repository hygiene",
                "No credentials, data/model binaries, agent files or decorative emoji in committed source",
            ],
            ["Bonus", "Optional Grad-CAM not attempted"],
        ],
        [145, 383],
    )
    p(
        "Before handing in, review this report, share both private repository links with the grader, and "
        "provide the API key through the instructor's private channel if authenticated prediction access "
        "is needed. Do not place it in the PDF. There is no instructor page limit currently specified."
    )
    h("Reproduction commands")
    code(
        ".venv/bin/python scripts/check_disk.py\n.venv/bin/dvc pull -j 2\n.venv/bin/python scripts/check_repository.py --history\n.venv/bin/python -m ruff check scripts visionguard tests docker cloud wsgi.py gunicorn.conf.py\n.venv/bin/python -m pytest -q\nsudo docker compose up -d --no-build --wait"
    )
    p(
        "dvc pull needs private access and sufficient space for missing objects; do not copy another full "
        "dataset on this constrained disk. CI builds on a fresh runner. The report builder consumes saved "
        "metrics and screenshots and performs no inference. The one-time final evaluator must not be "
        "rerun merely to regenerate figures. Consult docs/model-selection.md, docs/cloud-api.md and "
        "docs/cloud-integration.md for the model release and server-only settings."
    )
    h("References and evidence index")
    for label, url in [
        (
            "Assignment dataset",
            "https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images",
        ),
        ("Keras transfer learning", "https://keras.io/guides/transfer_learning/"),
        (
            "MLflow TensorFlow tracking",
            "https://mlflow.org/docs/latest/api_reference/python_api/mlflow.tensorflow.html",
        ),
        ("DVC remote storage", "https://doc.dvc.org/command-reference/remote/add"),
        (
            "Docker Compose services",
            "https://docs.docker.com/reference/compose-file/services/",
        ),
        (
            "GitHub Actions security",
            "https://docs.github.com/en/actions/reference/security/secure-use",
        ),
        ("Vercel Hobby", "https://vercel.com/docs/plans/hobby"),
        ("Vercel function limits", "https://vercel.com/docs/functions/limitations"),
        (
            "Postman local collection runs",
            "https://learning.postman.com/docs/postman-cli/postman-cli-run-collection",
        ),
    ]:
        p(f"{escape(label)}: <link href='{url}'>{url}</link>", "Reference")
    p(
        "Detailed task-to-file mapping: docs/assignment-checklist.md. Machine-readable receipts: "
        "reports/data, reports/training, reports/selection, reports/evaluation, reports/registry, "
        "reports/application, reports/container, reports/continuous-integration, reports/cloud, "
        "reports/integration and reports/labs. Figures are generated from those recorded results. "
        "Original command logs, database backups, secrets and runtime caches remain ignored."
    )

    output = ROOT / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "23L-0907_VisionGuard.pdf"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#bbcbd1"))
        canvas.line(42, 37, 570, 37)
        canvas.setFont("Body", 7)
        canvas.setFillColor(colors.HexColor("#50616a"))
        canvas.drawString(
            42,
            24,
            "Syed Muhammad Hanzala | 23L-0907_A2_Ops | " + now.strftime("%d %b %Y"),
        )
        canvas.drawRightString(570, 24, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(target),
        pagesize=(612, 792),
        leftMargin=42,
        rightMargin=42,
        topMargin=42,
        bottomMargin=52,
        title="23L-0907 VisionGuard assignment report",
        author="Syed Muhammad Hanzala",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    assert target.stat().st_size < 10 * 1024**2, (
        "Keep the report below the Git file-size limit"
    )
    assert shutil.disk_usage(ROOT).free >= 2 * 1024**3
    print(
        json.dumps(
            {
                "pdf": str(target),
                "source_commit": commit,
                "figures": figure_number,
                "bytes": target.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
