/* Dashboard: GPU and ingest-queue panels. */
(function (window, $, App) {
  "use strict";

  var POLL_MS = 5000;

  function missing(value) {
    return value === null || value === undefined;
  }

  function shown(value, suffix) {
    if (missing(value)) {
      return "—";
    }
    return suffix ? String(value) + suffix : String(value);
  }

  function meter(percent) {
    if (missing(percent)) {
      return $();
    }
    var width = Math.max(0, Math.min(100, Number(percent)));
    var $bar = $("<div>", { "class": "monitor-meter" });
    $("<span>").css("width", width + "%").appendTo($bar);
    return $bar;
  }

  function fillStats($list, rows) {
    $list.empty();
    rows.forEach(function (row) {
      $("<dt>").text(row.label).appendTo($list);
      var $value = $("<dd>").text(row.value);
      if (row.meter) {
        $value.append(row.meter);
      }
      $value.appendTo($list);
    });
  }

  function renderGpu(gpu) {
    var $state = $("#gpu-state");
    if (gpu.available) {
      $state.text("GPU online").removeClass("is-idle").addClass("is-ok");
    } else {
      $state.text("Running on CPU").removeClass("is-ok").addClass("is-idle");
    }
    var memory = "—";
    var memoryPct = null;
    if (!missing(gpu.memory_used_mb) && gpu.memory_total_mb) {
      memory = gpu.memory_used_mb + " / " + gpu.memory_total_mb + " MB";
      memoryPct = Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100);
    }
    fillStats($("#gpu-stats"), [
      { label: "Device", value: gpu.name || gpu.device || "—" },
      { label: "Utilization", value: shown(gpu.utilization_percent, "%"), meter: meter(gpu.utilization_percent) },
      { label: "Memory", value: memory, meter: meter(memoryPct) },
      { label: "Temperature", value: shown(gpu.temperature_c, "°C") }
    ]);
  }

  function renderQueue(queue) {
    var $state = $("#queue-state");
    if (!queue.broker_ok) {
      $state.text("Broker unreachable").removeClass("is-ok").addClass("is-bad");
    } else if ((queue.waiting || 0) + (queue.active || 0) + (queue.processing || 0) > 0) {
      $state.text("Jobs in progress").removeClass("is-bad").addClass("is-ok");
    } else {
      $state.text("Idle").removeClass("is-ok is-bad").addClass("is-idle");
    }
    fillStats($("#queue-stats"), [
      { label: "Waiting", value: shown(queue.waiting) },
      { label: "Active", value: shown(queue.active) },
      { label: "Workers", value: shown(queue.workers) },
      { label: "Your processing", value: shown(queue.processing) },
      { label: "Your failed", value: shown(queue.failed) }
    ]);
    var $items = $("#queue-list").empty();
    (queue.items || []).forEach(function (item) {
      var $row = $("<li>");
      $("<span>", { "class": "monitor-file" }).text(item.file_name).appendTo($row);
      $("<span>", { "class": "doc-status is-processing" }).text("processing").appendTo($row);
      $items.append($row);
    });
  }

  function refresh() {
    return App.api("GET", "/monitor")
      .done(function (data) {
        renderGpu(data.gpu || {});
        renderQueue(data.queue || {});
        $("#monitor-error").text("").removeClass("is-visible");
        $("#monitor-updated").text("Updated " + new Date().toLocaleTimeString());
      })
      .fail(function (xhr) {
        $("#monitor-error")
          .text(App.errorMessage(xhr, "Monitoring is unavailable."))
          .addClass("is-visible");
      });
  }

  App.startDashboard = function () {
    refresh();
    window.setInterval(function () {
      if (window.document.visibilityState === "hidden") {
        return;
      }
      refresh();
    }, POLL_MS);
  };
})(window, jQuery, window.App);
