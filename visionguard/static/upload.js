"use strict";
const input = document.getElementById("image");
const preview = document.getElementById("image-preview");
const status = document.getElementById("file-status");
let previewUrl;
if (input) {
  input.addEventListener("change", () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    preview.hidden = true;
    const file = input.files[0];
    status.textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(2)} MiB` : "No image selected.";
    if (file && ["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      previewUrl = URL.createObjectURL(file);
      preview.src = previewUrl;
      preview.hidden = false;
    }
  });
  preview.addEventListener("error", () => { preview.hidden = true; });
  document.getElementById("upload-form").addEventListener("submit", () => {
    document.getElementById("submit-button").disabled = true;
    document.getElementById("submit-status").textContent = "Checking your image. This may take a moment.";
  });
  window.addEventListener("pageshow", () => {
    document.getElementById("submit-button").disabled = false;
  });
}
