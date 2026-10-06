/* Profile memory: list, add, update, and delete for the signed-in user. */
(function (window, $, App) {
  "use strict";

  var TYPES = [
    "general",
    "personal",
    "professional",
    "technical",
    "preference",
    "project",
    "goal",
    "semantic",
    "episodic"
  ];

  function showAlert(message) {
    $("#memory-alert").text(message || "").toggleClass("is-visible", Boolean(message));
  }

  function typeOptions(selected) {
    return TYPES.map(function (type) {
      var mark = type === selected ? " selected" : "";
      return '<option value="' + type + '"' + mark + ">" + type + "</option>";
    }).join("");
  }

  function panelHtml() {
    return (
      '<div class="memory-backdrop" id="memory-backdrop" hidden>' +
        '<div class="memory-panel" role="dialog" aria-modal="true" aria-labelledby="memory-title">' +
          '<div class="memory-head">' +
            '<h2 id="memory-title">Memory</h2>' +
            '<button type="button" class="btn ghost" id="memory-close">Close</button>' +
          "</div>" +
          '<form id="memory-form">' +
            '<label>New memory' +
              '<textarea id="memory-text" rows="3" maxlength="500" required placeholder="A fact to remember"></textarea>' +
            "</label>" +
            '<label>Type' +
              '<select id="memory-type">' + typeOptions("general") + "</select>" +
            "</label>" +
            '<button type="submit" class="btn" id="memory-add">Add memory</button>' +
          "</form>" +
          '<div class="alert" id="memory-alert" role="alert"></div>' +
          '<ul class="memory-list" id="memory-list"></ul>' +
        "</div>" +
      "</div>"
    );
  }

  function renderList(items) {
    var $list = $("#memory-list").empty();
    if (!items.length) {
      $list.append('<li class="memory-empty">No memories yet.</li>');
      return;
    }
    items.forEach(function (item) {
      var $li = $('<li class="memory-item"></li>').attr("data-id", item.id);
      $li.append($('<p class="memory-text"></p>').text(item.memory));
      $li.append($('<span class="memory-type"></span>').text(item.memory_type));
      var $actions = $('<div class="memory-actions"></div>');
      $actions.append('<button type="button" class="btn ghost" data-edit>Update</button>');
      $actions.append('<button type="button" class="btn ghost memory-delete" data-delete>Delete</button>');
      $li.append($actions);
      $list.append($li);
    });
  }

  function loadMemories() {
    return App.api("GET", "/memories")
      .done(function (items) {
        renderList(items || []);
      })
      .fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not load memories."));
      });
  }

  function openPanel() {
    showAlert("");
    $("#memory-backdrop").prop("hidden", false);
    loadMemories();
    $("#memory-text").focus();
  }

  function closePanel() {
    $("#memory-backdrop").prop("hidden", true);
  }

  function startEdit($item) {
    var id = $item.data("id");
    var text = $item.find(".memory-text").text();
    var type = $item.find(".memory-type").text();
    $item.empty();
    var $form = $('<form class="memory-edit"></form>');
    $form.append(
      $('<textarea rows="3" maxlength="500" required></textarea>').val(text)
    );
    $form.append($("<select></select>").html(typeOptions(type)).val(type));
    var $actions = $('<div class="memory-actions"></div>');
    $actions.append('<button type="submit" class="btn">Save</button>');
    $actions.append('<button type="button" class="btn ghost" data-cancel>Cancel</button>');
    $form.append($actions);
    $form.on("submit", function (event) {
      event.preventDefault();
      var next = $form.find("textarea").val().trim();
      if (!next) {
        return;
      }
      App.api.json("PUT", "/memories/" + encodeURIComponent(id), {
        memory: next,
        memory_type: $form.find("select").val()
      }).done(function () {
        showAlert("");
        loadMemories();
      }).fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not update that memory."));
      });
    });
    $item.append($form);
  }

  function bindPanel() {
    if (!$("#username").length || $("#memory-backdrop").length) {
      return;
    }
    $("body").append(panelHtml());

    $("#username").on("click", openPanel);
    $("#memory-close").on("click", closePanel);
    $("#memory-backdrop").on("click", function (event) {
      if (event.target === this) {
        closePanel();
      }
    });

    $("#memory-form").on("submit", function (event) {
      event.preventDefault();
      var text = $("#memory-text").val().trim();
      if (!text) {
        return;
      }
      $("#memory-add").prop("disabled", true);
      App.api.json("POST", "/memories", {
        memory: text,
        memory_type: $("#memory-type").val()
      }).done(function () {
        $("#memory-text").val("");
        $("#memory-type").val("general");
        showAlert("");
        loadMemories();
      }).fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not add that memory."));
      }).always(function () {
        $("#memory-add").prop("disabled", false);
      });
    });

    $("#memory-list").on("click", "[data-edit]", function () {
      startEdit($(this).closest(".memory-item"));
    });

    $("#memory-list").on("click", "[data-cancel]", function () {
      loadMemories();
    });

    $("#memory-list").on("click", "[data-delete]", function () {
      var id = $(this).closest(".memory-item").data("id");
      App.api("DELETE", "/memories/" + encodeURIComponent(id), { dataType: "text" })
        .done(function () {
          showAlert("");
          loadMemories();
        })
        .fail(function (xhr) {
          showAlert(App.errorMessage(xhr, "Could not delete that memory."));
        });
    });
  }

  $(bindPanel);
})(window, jQuery, window.App);
