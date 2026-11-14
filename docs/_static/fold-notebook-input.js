// Wrap tagged inputs only. Notebook output stays outside the disclosure.
for (const marker of document.querySelectorAll(".fold-notebook-input")) {
    const input = marker.nextElementSibling;
    if (!input?.classList.contains("nbinput")) continue;

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = marker.dataset.summary;
    marker.replaceWith(details);
    details.append(summary, input);
}
