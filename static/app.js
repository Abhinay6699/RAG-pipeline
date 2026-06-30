/**
 * app.js — Frontend logic for the Document QA system.
 *
 * Handles file upload, document list refresh, query submission,
 * and citation rendering with expand/collapse.
 */

(function () {
    "use strict";

    // ── DOM elements ──────────────────────────────────────────────
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const fileLabel = document.getElementById("file-label");
    const fileLabelText = document.getElementById("file-label-text");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadFeedback = document.getElementById("upload-feedback");

    const documentList = document.getElementById("document-list");
    const noDocsMsg = document.getElementById("no-docs-msg");

    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const queryBtn = document.getElementById("query-btn");
    const resultsArea = document.getElementById("results-area");
    const welcomeMsg = document.getElementById("welcome-msg");

    const statusDot = document.querySelector(".status-dot");
    const statusText = document.getElementById("status-text");

    // ── Health check ──────────────────────────────────────────────

    async function checkHealth() {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            statusDot.classList.add("online");
            statusDot.classList.remove("error");
            statusText.textContent = data.indexed_chunks + " chunks indexed";
        } catch {
            statusDot.classList.add("error");
            statusDot.classList.remove("online");
            statusText.textContent = "Offline";
        }
    }

    // ── Load document list ────────────────────────────────────────

    async function loadDocuments() {
        try {
            const res = await fetch("/documents");
            const data = await res.json();

            if (data.documents.length === 0) {
                documentList.innerHTML = "";
                documentList.appendChild(noDocsMsg);
                noDocsMsg.style.display = "block";
                return;
            }

            noDocsMsg.style.display = "none";
            documentList.innerHTML = "";

            data.documents.forEach(function (doc) {
                var item = document.createElement("div");
                item.className = "doc-item";
                item.innerHTML =
                    '<span class="doc-name" title="' + escapeHtml(doc.filename) + '">' +
                    escapeHtml(doc.filename) +
                    '</span>' +
                    '<span class="doc-chunks">' + doc.chunk_count + ' chunks</span>';
                documentList.appendChild(item);
            });
        } catch (err) {
            console.error("Failed to load documents:", err);
        }
    }

    // ── File upload ───────────────────────────────────────────────

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            fileLabelText.textContent = fileInput.files[0].name;
            uploadBtn.disabled = false;
        } else {
            fileLabelText.textContent = "Choose PDF, TXT, or DOCX";
            uploadBtn.disabled = true;
        }
    });

    uploadForm.addEventListener("submit", async function (e) {
        e.preventDefault();

        if (!fileInput.files.length) return;

        uploadBtn.disabled = true;
        setFeedback("Uploading and indexing...", "loading");

        var formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            var res = await fetch("/upload", {
                method: "POST",
                body: formData,
            });
            var data = await res.json();

            if (!res.ok) {
                setFeedback(data.error || "Upload failed.", "error");
                uploadBtn.disabled = false;
                return;
            }

            setFeedback(
                data.filename + " indexed (" + data.chunk_count + " chunks)",
                "success"
            );

            // Reset form
            fileInput.value = "";
            fileLabelText.textContent = "Choose PDF, TXT, or DOCX";
            uploadBtn.disabled = true;

            // Refresh lists
            loadDocuments();
            checkHealth();
        } catch (err) {
            setFeedback("Network error: " + err.message, "error");
            uploadBtn.disabled = false;
        }
    });

    function setFeedback(msg, type) {
        uploadFeedback.textContent = msg;
        uploadFeedback.className = "upload-feedback " + type;
    }

    // ── Query submission ──────────────────────────────────────────

    queryForm.addEventListener("submit", async function (e) {
        e.preventDefault();

        var question = queryInput.value.trim();
        if (!question) return;

        // Hide welcome message
        if (welcomeMsg) welcomeMsg.style.display = "none";

        // Show user query
        var entry = document.createElement("div");
        entry.className = "result-entry";

        var bubble = document.createElement("div");
        bubble.className = "query-bubble";
        bubble.textContent = question;
        entry.appendChild(bubble);

        // Loading indicator
        var loading = document.createElement("div");
        loading.className = "loading-indicator";
        loading.innerHTML =
            '<div class="loading-dots"><span></span><span></span><span></span></div>' +
            '<span>Searching and generating answer...</span>';
        entry.appendChild(loading);

        resultsArea.appendChild(entry);
        resultsArea.scrollTop = resultsArea.scrollHeight;

        queryInput.value = "";
        queryBtn.disabled = true;

        try {
            var res = await fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question }),
            });
            var data = await res.json();

            // Remove loading
            entry.removeChild(loading);

            if (!res.ok) {
                var errDiv = document.createElement("div");
                errDiv.className = "error-msg";
                errDiv.textContent = data.error || "Query failed.";
                entry.appendChild(errDiv);
                return;
            }

            // Render answer
            renderAnswer(entry, data);
        } catch (err) {
            entry.removeChild(loading);
            var errDiv = document.createElement("div");
            errDiv.className = "error-msg";
            errDiv.textContent = "Network error: " + err.message;
            entry.appendChild(errDiv);
        } finally {
            queryBtn.disabled = false;
            resultsArea.scrollTop = resultsArea.scrollHeight;
        }
    });

    // ── Render answer + citations ─────────────────────────────────

    function renderAnswer(entry, data) {
        // Answer text
        var answerBlock = document.createElement("div");
        answerBlock.className = "answer-block";

        var paragraphs = data.answer.split("\n").filter(function (p) {
            return p.trim().length > 0;
        });
        paragraphs.forEach(function (p) {
            var para = document.createElement("p");
            para.textContent = p;
            answerBlock.appendChild(para);
        });
        entry.appendChild(answerBlock);

        // Model meta
        var meta = document.createElement("div");
        meta.className = "answer-meta";
        meta.textContent = "Model: " + data.model;
        if (data.usage && data.usage.total_tokens) {
            meta.textContent += " | " + data.usage.total_tokens + " tokens";
        }
        entry.appendChild(meta);

        // Citations
        if (data.citations && data.citations.length > 0) {
            var citBar = document.createElement("div");
            citBar.className = "citations-bar";

            var detailContainer = document.createElement("div");

            data.citations.forEach(function (cit, idx) {
                // Tag
                var tag = document.createElement("span");
                tag.className = "citation-tag";
                tag.textContent =
                    cit.source + " #" + cit.chunk_index +
                    " [" + (cit.similarity_score || 0).toFixed(2) + "]";
                tag.setAttribute("data-idx", idx);

                // Detail panel
                var detail = document.createElement("div");
                detail.className = "citation-detail";
                detail.id = "citation-detail-" + Date.now() + "-" + idx;

                detail.innerHTML =
                    '<div class="citation-detail-header">' +
                    '<span class="citation-source">' + escapeHtml(cit.source) +
                    ' — chunk #' + cit.chunk_index + '</span>' +
                    '<span class="citation-score">score: ' +
                    (cit.similarity_score || 0).toFixed(4) +
                    ' | rrf: ' + (cit.rrf_score || 0).toFixed(4) +
                    '</span></div>' +
                    '<div class="citation-excerpt">' + escapeHtml(cit.excerpt) + '</div>';

                // Toggle on click
                tag.addEventListener("click", function () {
                    var isVisible = detail.classList.contains("visible");
                    // Close all others in this entry
                    var allDetails = detailContainer.querySelectorAll(".citation-detail");
                    var allTags = citBar.querySelectorAll(".citation-tag");
                    allDetails.forEach(function (d) { d.classList.remove("visible"); });
                    allTags.forEach(function (t) { t.classList.remove("active"); });

                    if (!isVisible) {
                        detail.classList.add("visible");
                        tag.classList.add("active");
                    }
                });

                citBar.appendChild(tag);
                detailContainer.appendChild(detail);
            });

            entry.appendChild(citBar);
            entry.appendChild(detailContainer);
        }
    }

    // ── Utilities ─────────────────────────────────────────────────

    function escapeHtml(text) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    // ── Init ──────────────────────────────────────────────────────

    checkHealth();
    loadDocuments();

    // Refresh health every 30s
    setInterval(checkHealth, 30000);
})();
