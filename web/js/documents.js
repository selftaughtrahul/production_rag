/* Document list for the signed-in user. */
(function (window, $, App) {
  "use strict";

  function showAlert(message) {
    $("#docs-alert").text(message || "").toggleClass("is-visible", Boolean(message));
  }

  function formatWhen(value) {
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "";
    }
    return date.toLocaleString();
  }

  function renderList(items) {
    var $list = $("#doc-list").empty();
    if (!items || !items.length) {
      $list.append('<li class="doc-empty">No documents yet.</li>');
      return;
    }
    items.forEach(function (item) {
      var $li = $('<li class="doc-item"></li>');
      $li.append($('<span class="doc-name"></span>').text(item.file_name || "Untitled"));
      $li.append($('<span class="doc-when"></span>').text(formatWhen(item.created_at)));
      $list.append($li);
    });
  }

  App.startDocuments = function () {
    App.api("GET", "/documents")
      .done(function (items) {
        showAlert("");
        renderList(items);
      })
      .fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not load documents."));
      });
  };
})(window, jQuery, window.App);
