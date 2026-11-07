"use strict";

const form = document.querySelector("#image-form");
const input = document.querySelector("#image-input");
const area = document.querySelector("#drop-area");
const label = document.querySelector(".file-label");
const previewArea = document.querySelector("#preview-area");
const preview = document.querySelector("#preview");
const button = document.querySelector("#check-button");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const result = document.querySelector("#result");
let previewUrl = null;
let busy = false;
let selection = 0;

function showError(message) {
  error.textContent = message;
  error.hidden = false;
}

function clearPreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.removeAttribute("src");
  previewArea.hidden = true;
  label.hidden = false;
}

input.addEventListener("change", () => {
  const current = ++selection;
  const file = input.files[0];
  result.hidden = true;
  error.hidden = true;
  button.disabled = true;
  clearPreview();
  if (!file) {
    status.textContent = "Choose an image to get started.";
    return;
  }
  if (!/\.(jpe?g|png|webp)$/i.test(file.name) || (file.type && !["image/jpeg", "image/png", "image/webp"].includes(file.type))) {
    showError("Choose a JPEG, PNG or WebP image.");
    status.textContent = "This file type is not supported.";
    return;
  }
  if (file.size === 0 || file.size > 4 * 1024 * 1024) {
    showError(file.size === 0 ? "This image file is empty." : "Choose an image smaller than 4 MiB.");
    status.textContent = "Choose a different image to continue.";
    return;
  }
  const url = URL.createObjectURL(file);
  previewUrl = url;
  const candidate = new Image();
  status.textContent = "Preparing preview...";
  candidate.onload = () => {
    if (current !== selection) return;
    if (candidate.naturalWidth * candidate.naturalHeight > 4000000) {
      clearPreview();
      showError("Resize this image to at most 4 million pixels, then try again.");
      status.textContent = "Choose a smaller image to continue.";
      return;
    }
    preview.src = url;
    document.querySelector("#filename").textContent = file.name;
    previewArea.hidden = false;
    label.hidden = true;
    button.disabled = busy;
    status.textContent = "Ready to check. Your image will be sent for classification.";
  };
  candidate.onerror = () => {
    if (current !== selection) return;
    clearPreview();
    showError("This file could not be opened as an image. Choose a different file.");
    status.textContent = "Choose a different image to continue.";
  };
  candidate.src = url;
});

for (const eventName of ["dragenter", "dragover"]) {
  area.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!busy) area.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  area.addEventListener(eventName, (event) => {
    event.preventDefault();
    area.classList.remove("dragging");
  });
}
area.addEventListener("drop", (event) => {
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) {
    showError("Choose one image at a time.");
    return;
  }
  input.files = event.dataTransfer.files;
  input.dispatchEvent(new Event("change"));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || button.disabled || !input.files[0]) return;
  busy = true;
  input.disabled = true;
  button.disabled = true;
  button.textContent = "Checking image...";
  status.textContent = "Looking at visual patterns. This may take a few seconds.";
  error.hidden = true;
  result.hidden = true;
  form.setAttribute("aria-busy", "true");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 55000);
  const body = new FormData(form);
  body.set("image", input.files[0]);
  try {
    const response = await fetch(form.action, {
      method: "POST", body,
      headers: { "Accept": "application/json" },
      credentials: "same-origin",
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const fallback = response.status === 413 ? "This image is too large. Choose a smaller file." : response.status === 429 ? "Please wait a minute before trying again." : response.status === 503 ? "The service is busy. Please try again in a moment." : "The image could not be checked. Please try again.";
      throw new Error(data?.message || fallback);
    }
    if (!data || !["real", "ai_generated"].includes(data.label) || !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1) {
      throw new Error("The service returned an incomplete result. Please try again.");
    }
    const isAi = data.label === "ai_generated";
    document.querySelector("#result-title").textContent = isAi ? "Likely AI-generated" : "Likely real";
    document.querySelector("#confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.querySelector("#confidence-meter").value = data.confidence;
    document.querySelector("#result-description").textContent = "Consider the image's source and context alongside this estimate.";
    result.hidden = false;
    status.textContent = "Check complete. Choose another image whenever you're ready.";
    document.querySelector("#result-title").focus({ preventScroll: true });
    result.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
  } catch (failure) {
    showError(failure.name === "AbortError" ? "The check took too long. Please try again in a moment." : failure instanceof TypeError ? "The service could not be reached. Check your connection and try again." : failure.message);
    status.textContent = "Your image is still selected. You can try again.";
  } finally {
    clearTimeout(timeout);
    busy = false;
    input.disabled = false;
    button.disabled = false;
    button.textContent = "Check image";
    form.removeAttribute("aria-busy");
  }
});

window.addEventListener("pagehide", clearPreview);
