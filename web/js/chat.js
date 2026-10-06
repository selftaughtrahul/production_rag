/* ChatGPT-style thread: left history by latest, click to load, SSE replies. */
(function (window, $, App) {
  "use strict";

  var sessionId = null;
  var busy = false;

  function escapeHtml(value) {
    return $("<div>").text(value == null ? "" : String(value)).html();
  }

  function messageText(content) {
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return content.map(function (part) {
        if (typeof part === "string") {
          return part;
        }
        if (part && typeof part.text === "string") {
          return part.text;
        }
        return "";
      }).join("");
    }
    return content == null ? "" : String(content);
  }

  function normalizeRole(role) {
    var value = String(role || "").toLowerCase();
    if (value === "human" || value === "user") {
      return "user";
    }
    if (value === "ai" || value === "assistant") {
      return "assistant";
    }
    return "";
  }

  function titleFrom(text) {
    var clean = String(text || "").replace(/\s+/g, " ").trim();
    if (!clean) {
      return "New chat";
    }
    return clean.length > 42 ? clean.slice(0, 42) + "…" : clean;
  }

  function showAlert(message) {
    $("#chat-alert").text(message || "").toggleClass("is-visible", Boolean(message));
  }

  function setBusy(on) {
    busy = on;
    $("#send").prop("disabled", on);
    $("#question").prop("disabled", on);
  }

  function emptyState() {
    return $(
      '<div class="gpt-empty" id="empty-state">' +
        "<h1>Where should we start?</h1>" +
        "<p>Ask about your documents. History on the left opens the full thread.</p>" +
      "</div>"
    );
  }

  function bubble(role, text) {
    var who = role === "user" ? "You" : "Assistant";
    var $row = $('<div class="gpt-row"></div>').addClass("is-" + role);
    $row.append($('<div class="gpt-who"></div>').text(who));
    $row.append($('<div class="gpt-bubble"></div>').html(escapeHtml(text)));
    return $row;
  }

  function clearThread() {
    $("#thread").empty().append(emptyState());
    $("#chat-title").text("New chat");
  }

  function appendMessage(role, text) {
    $("#empty-state").remove();
    var $row = bubble(role, text);
    $("#thread").append($row);
    $("#thread").scrollTop($("#thread")[0].scrollHeight);
    return $row.find(".gpt-bubble");
  }

  function markActive() {
    $("#conversation-list .gpt-conv").removeClass("is-active");
    if (sessionId) {
      $('#conversation-list .gpt-conv[data-session="' + sessionId + '"]').addClass("is-active");
    }
  }

  function renderList(sessions) {
    var $list = $("#conversation-list").empty();
    if (!sessions || !sessions.length) {
      $list.append('<p class="gpt-none">No chats yet</p>');
      return;
    }
    sessions.forEach(function (item) {
      var id = item.session_id;
      var label = titleFrom(item.first_message);
      var $btn = $('<button type="button" class="gpt-conv"></button>')
        .attr("data-session", id)
        .attr("title", item.first_message || id)
        .append($('<span class="gpt-conv-title"></span>').text(label))
        .append($('<span class="gpt-conv-del" data-delete="' + id + '" aria-label="Delete">×</span>'));
      $list.append($btn);
    });
    markActive();
  }

  function refreshList() {
    return App.api("GET", "/chat/conversations")
      .done(function (res) {
        renderList(res.sessions || []);
      })
      .fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not load chats."));
      });
  }

  function openThread(id) {
    if (busy || !id) {
      return;
    }
    showAlert("");
    App.api("GET", "/chat/conversations/" + encodeURIComponent(id))
      .done(function (res) {
        sessionId = res.session_id;
        $("#thread").empty();
        var messages = res.messages || [];
        var title = "New chat";
        messages.forEach(function (msg) {
          var role = normalizeRole(msg.role);
          var text = messageText(msg.content);
          if (!role || !text) {
            return;
          }
          // Specialist nodes store notes as HumanMessages like "[RAG Specialist]: ..."
          if (/^\[[^\]]+\]:/.test(text.trim())) {
            return;
          }
          appendMessage(role, text);
          if (role === "user" && title === "New chat") {
            title = titleFrom(text);
          }
        });
        if (!$("#thread").children().length) {
          clearThread();
        }
        $("#chat-title").text(title);
        markActive();
        $("#thread").scrollTop($("#thread")[0].scrollHeight);
      })
      .fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not open that chat."));
      });
  }

  function newChat() {
    if (busy) {
      return;
    }
    sessionId = null;
    showAlert("");
    clearThread();
    markActive();
    $("#question").focus();
  }

  function deleteThread(id) {
    if (busy || !id) {
      return;
    }
    App.api("DELETE", "/chat/conversations/" + encodeURIComponent(id))
      .done(function () {
        if (sessionId === id) {
          newChat();
        }
        refreshList();
      })
      .fail(function (xhr) {
        showAlert(App.errorMessage(xhr, "Could not delete that chat."));
      });
  }

  function sendQuestion(question) {
    setBusy(true);
    showAlert("");
    $("#empty-state").remove();
    appendMessage("user", question);
    var $answer = appendMessage("assistant", "");
    var body = { question: question };
    if (sessionId) {
      body.session_id = sessionId;
    }

    var seen = "";
    $answer.addClass("is-streaming");
    return App.api.stream("/chat/", body, function (event) {
      if (!event || !event.type) {
        return;
      }
      if (event.type === "token") {
        seen += event.content || "";
        $answer.text(seen);
        $("#thread").scrollTop($("#thread")[0].scrollHeight);
      } else if (event.type === "done") {
        sessionId = event.session_id || sessionId;
        $answer.removeClass("is-streaming");
        if (event.answer && event.answer !== seen) {
          $answer.text(event.answer);
        }
        $("#chat-title").text(titleFrom(question));
        refreshList();
      } else if (event.type === "error") {
        $answer.removeClass("is-streaming");
        $answer.text(event.error || "Query could not be completed.");
      }
    }).catch(function (err) {
      var message = (err && err.message) || "Query could not be completed.";
      $answer.removeClass("is-streaming");
      $answer.text(message);
      showAlert(message);
    }).finally(function () {
      setBusy(false);
      $("#question").val("").trigger("input").focus();
    });
  }

  function growComposer() {
    var el = $("#question")[0];
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  App.startChat = function () {
    refreshList();
    newChat();

    $("#new-chat").on("click", newChat);
    $("#toggle-sidebar").on("click", function () {
      $("#sidebar").toggleClass("is-open");
    });

    $("#conversation-list").on("click", ".gpt-conv", function (event) {
      if ($(event.target).closest("[data-delete]").length) {
        return;
      }
      openThread($(this).data("session"));
    });

    $("#conversation-list").on("click", "[data-delete]", function (event) {
      event.preventDefault();
      event.stopPropagation();
      deleteThread($(this).data("delete"));
    });

    $("#question").on("input", growComposer);
    $("#question").on("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        $("#chat-form").submit();
      }
    });

    $("#chat-form").on("submit", function (event) {
      event.preventDefault();
      var question = $("#question").val().trim();
      if (!question || busy) {
        return;
      }
      sendQuestion(question);
    });
  };
})(window, jQuery, window.App);
