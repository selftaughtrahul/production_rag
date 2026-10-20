/* Documents page: upload, grid/list views, download, delete. */
(function (window, $, App) {
  "use strict";

  var VIEW_KEY = "rag_docs_view";
  var items = [];
  var openId = null;
  var previewFor = null;
  var pollTimer = null;

  var STATUS_LABEL = {
    processing: "Processing",
    ready: "Ready",
    failed: "Failed"
  };

  function showAlert(message) {
    $("#docs-alert").text(message || "").toggleClass("is-visible", Boolean(message));
  }

  function showInfo(message) {
    $("#docs-info").text(message || "").toggleClass("is-visible", Boolean(message));
  }

  function formatWhen(value) {
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "";
    }
    return date.toLocaleString();
  }

  function formatSize(bytes) {
    if (bytes == null) {
      return "File missing";
    }
    if (bytes < 1024) {
      return bytes + " B";
    }
    var units = ["KB", "MB", "GB"];
    var value = bytes / 1024;
    var i = 0;
    while (value >= 1024 && i < units.length - 1) {
      value /= 1024;
      i += 1;
    }
    return value.toFixed(value >= 10 ? 0 : 1) + " " + units[i];
  }

  function shortId(id) {
    return String(id || "").slice(0, 8);
  }

  function getView() {
    var saved = window.localStorage.getItem(VIEW_KEY);
    return saved === "grid" ? "grid" : "list";
  }

  function setView(view) {
    window.localStorage.setItem(VIEW_KEY, view);
    $("#doc-list").toggleClass("is-grid", view === "grid").toggleClass("is-list", view !== "grid");
    $(".view-btn").each(function () {
      var active = $(this).data("view") === view;
      $(this).toggleClass("is-active", active).attr("aria-pressed", active ? "true" : "false");
    });
  }

  function metaItem(label, value) {
    return $('<span class="doc-meta-item"></span>')
      .append($('<span class="doc-meta-label"></span>').text(label))
      .append($('<span class="doc-meta-value"></span>').text(value));
  }

  function statusOf(item) {
    return item && STATUS_LABEL[item.status] ? item.status : "ready";
  }

  function statusText(item) {
    var status = statusOf(item);
    var label = STATUS_LABEL[status];
    if (item && item.error_message) {
      return label + " — " + item.error_message;
    }
    return label;
  }

  function setProgress(percent, label, processing) {
    var $box = $("#upload-progress").prop("hidden", false);
    $box.toggleClass("is-processing", Boolean(processing));
    $("#upload-progress-bar").css("width", processing ? "" : percent + "%");
    $("#upload-progress-label").text(label || "");
  }

  function hideProgress() {
    $("#upload-progress").prop("hidden", true).removeClass("is-processing");
    $("#upload-progress-bar").css("width", "0");
    $("#upload-progress-label").text("");
  }

  function renderItem(item) {
    var status = statusOf(item);
    var $li = $('<li class="doc-item"></li>').attr("data-id", item.id);

    var $head = $('<div class="doc-head"></div>');
    $head.append(
      $('<button type="button" class="doc-name" data-action="show"></button>')
        .text(item.file_name || "Untitled")
        .attr("title", item.file_name || "")
    );
    $head.append($('<span class="doc-status"></span>').addClass("is-" + status).text(STATUS_LABEL[status]));
    if (item.error_message) {
      $head.append($('<span class="doc-error"></span>').text(item.error_message));
    }
    $li.append($head);

    var $metaChips = $('<div class="doc-tags"></div>');
    if (item.metadata && Object.keys(item.metadata).length) {
      $metaChips.append(App.renderMeta(item.metadata));
    } else {
      $metaChips.append($('<span class="doc-no-tags"></span>').text("No metadata"));
    }
    $li.append($metaChips);

    var $meta = $('<div class="doc-meta"></div>');
    $meta.append(metaItem("Uploaded", formatWhen(item.created_at)));
    $meta.append(metaItem("Size", formatSize(item.file_size)));
    $meta.append(metaItem("Chunks", String(item.chunk_count == null ? 0 : item.chunk_count)));
    $meta.append(metaItem("ID", shortId(item.id)).attr("title", item.id));
    $li.append($meta);

    var $actions = $('<div class="doc-actions"></div>');
    $actions.append($('<button type="button" class="btn ghost small" data-action="show">Details</button>'));
    $actions.append(
      $('<button type="button" class="btn ghost small" data-action="download">Download</button>')
        .prop("disabled", item.file_size == null)
    );
    $actions.append($('<button type="button" class="btn ghost small danger" data-action="delete">Delete</button>'));
    $li.append($actions);

    return $li;
  }

  function renderList() {
    var $list = $("#doc-list").empty();
    $("#doc-count").text(items.length ? items.length + (items.length === 1 ? " document" : " documents") : "");
    if (!items.length) {
      $list.append('<li class="doc-empty">No documents yet.</li>');
      return;
    }
    items.forEach(function (item) {
      $list.append(renderItem(item));
    });
  }

  function load() {
    return App.api("GET", "/documents")
      .done(function (res) {
        items = res || [];
        showAlert("");
        renderList();
        syncOpenStatus();
        schedulePoll();
      })
      .fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not load documents."));
      });
  }

  function findItem(id) {
    for (var i = 0; i < items.length; i += 1) {
      if (items[i].id === id) {
        return items[i];
      }
    }
    return null;
  }

  /* The JWT lives in localStorage, so a plain <a href> would be unauthorized. */
  function download(id) {
    var item = findItem(id);
    if (!item) {
      return;
    }
    var headers = {};
    var token = App.getToken();
    if (token) {
      headers.Authorization = "Bearer " + token;
    }
    showAlert("");
    fetch("/documents/download/" + encodeURIComponent(id), { headers: headers })
      .then(function (res) {
        if (res.status === 401 || res.status === 403) {
          App.clearToken();
          window.location.replace("/login");
          throw new Error("Session expired.");
        }
        if (!res.ok) {
          return res.json().catch(function () { return {}; }).then(function (data) {
            throw new Error(typeof data.detail === "string" ? data.detail : "Download failed.");
          });
        }
        return res.blob();
      })
      .then(function (blob) {
        var url = window.URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = item.file_name || "document";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.setTimeout(function () { window.URL.revokeObjectURL(url); }, 1000);
      })
      .catch(function (err) {
        showAlert((err && err.message) || "Download failed.");
      });
  }

  function remove(id) {
    var item = findItem(id);
    if (!item) {
      return;
    }
    var ok = window.confirm(
      'Delete "' + (item.file_name || "this document") + '"?\n\n' +
      "This removes the file, its record, and all of its search vectors."
    );
    if (!ok) {
      return;
    }
    var $item = $('#doc-list .doc-item[data-id="' + id + '"]').addClass("is-busy");
    $item.find("button").prop("disabled", true);
    showAlert("");
    App.api("DELETE", "/documents/" + encodeURIComponent(id))
      .done(function (res) {
        var n = res && res.deleted_chunks != null ? res.deleted_chunks : 0;
        showInfo("Deleted " + (item.file_name || "document") + " (" + n + " chunks removed).");
        load();
      })
      .fail(function (xhr) {
        $item.removeClass("is-busy").find("button").prop("disabled", false);
        showAlert(App.errorMessage(xhr, "Could not delete that document."));
      });
  }

  function schedulePoll() {
    if (pollTimer) {
      return;
    }
    var pending = items.some(function (item) { return statusOf(item) === "processing"; });
    if (!pending) {
      return;
    }
    pollTimer = window.setTimeout(function () {
      pollTimer = null;
      load();
    }, 2000);
  }

  function closeDetail() {
    openId = null;
    previewFor = null;
    $("#doc-detail").prop("hidden", true);
  }

  function syncOpenStatus() {
    if (!openId) {
      return;
    }
    var item = findItem(openId);
    if (!item) {
      closeDetail();
      return;
    }
    $("#doc-detail-status").text(statusText(item));
    if (statusOf(item) === "ready" && previewFor !== item.id) {
      previewFor = item.id;
      App.api("GET", "/documents/" + encodeURIComponent(item.id))
        .done(function (doc) {
          if (openId === doc.id) {
            $("#doc-preview").text(doc.preview || "No preview.");
          }
        });
    }
  }

  function showDocument(id) {
    showAlert("");
    App.api("GET", "/documents/" + encodeURIComponent(id))
      .done(function (doc) {
        openId = doc.id;
        previewFor = statusOf(doc) === "ready" ? doc.id : null;
        $("#doc-detail").prop("hidden", false);
        $("#doc-detail-name").text(doc.file_name || "Untitled");
        $("#doc-detail-status").text(statusText(doc));
        var $detailMeta = $("#detail-meta").empty();
        if (doc.metadata && Object.keys(doc.metadata).length) {
          $detailMeta.append(App.renderMeta(doc.metadata));
        } else {
          $detailMeta.append($('<span class="doc-no-tags"></span>').text("No metadata yet"));
        }
        if (doc.preview) {
          $("#doc-preview").text(doc.preview);
        } else if (statusOf(doc) === "processing") {
          $("#doc-preview").text("Still processing…");
        } else {
          $("#doc-preview").text("No preview.");
        }
      })
      .fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not open that document."));
      });
  }

  /* Async (Celery) mode returns a task id; poll until it settles, then reload. */
  function pollTask(taskId, attempt) {
    attempt = attempt || 0;
    if (attempt > 60) {
      showInfo("Ingestion is still running. Refresh in a moment.");
      return;
    }
    App.api("GET", "/tasks/" + encodeURIComponent(taskId))
      .done(function (res) {
        if (res.status === "SUCCESS") {
          hideProgress();
          showInfo("Document ingested.");
          load();
        } else if (res.status === "FAILURE") {
          hideProgress();
          showInfo("");
          showAlert(res.error || "Ingestion failed.");
          load();
        } else {
          window.setTimeout(function () { pollTask(taskId, attempt + 1); }, 2000);
        }
      })
      .fail(function () {
        window.setTimeout(function () { pollTask(taskId, attempt + 1); }, 3000);
      });
  }

  function upload(event) {
    event.preventDefault();
    var input = $("#upload-file")[0];
    var file = input && input.files && input.files[0];
    if (!file) {
      showAlert("Choose a file to upload.");
      return;
    }
    var form = new FormData();
    form.append("file", file, file.name);

    showAlert("");
    showInfo("");
    setProgress(0, "Uploading " + file.name + "…", false);
    $("#upload-btn").prop("disabled", true);

    App.api("POST", "/ingest", {
      data: form,
      processData: false,
      contentType: false,
      xhr: function () {
        var xhr = $.ajaxSettings.xhr();
        if (xhr.upload) {
          xhr.upload.addEventListener("progress", function (event) {
            if (!event.lengthComputable) {
              return;
            }
            var pct = Math.round((event.loaded / event.total) * 100);
            setProgress(pct, "Uploading " + file.name + "… " + pct + "%", false);
          });
        }
        return xhr;
      }
    })
      .done(function (res) {
        $("#upload-file").val("");
        if (res && (res.task_id || res.status === "processing")) {
          setProgress(100, "Processing " + file.name + "…", true);
          showInfo("Upload received. Ingesting in the background…");
          load();
          if (res.task_id) {
            pollTask(res.task_id, 0);
          }
        } else {
          var n = res && res.ingestion && res.ingestion.chunks_count;
          hideProgress();
          showInfo("Ingested " + file.name + (n != null ? " (" + n + " chunks)." : "."));
          load();
        }
      })
      .fail(function (xhr) {
        hideProgress();
        showInfo("");
        showAlert(App.errorMessage(xhr, "Upload failed."));
        load();
      })
      .always(function () {
        $("#upload-btn").prop("disabled", false);
      });
  }

  App.startDocuments = function () {
    setView(getView());

    $(".view-btn").on("click", function () {
      setView($(this).data("view"));
    });

    $("#upload-form").on("submit", upload);

    $("#doc-list").on("click", "[data-action]", function () {
      var id = $(this).closest(".doc-item").data("id");
      var action = $(this).data("action");
      if (action === "download") {
        download(id);
      } else if (action === "delete") {
        remove(id);
      } else if (action === "show") {
        showDocument(id);
      }
    });

    $("#doc-detail-close").on("click", closeDetail);

    load();
  };
})(window, jQuery, window.App);
