/* Shared API helper: JWT storage plus authenticated jQuery AJAX. */
(function (window, $) {
  "use strict";

  var TOKEN_KEY = "rag_token";

  function getToken() {
    return window.localStorage.getItem(TOKEN_KEY);
  }

  function setToken(token) {
    window.localStorage.setItem(TOKEN_KEY, token);
  }

  function clearToken() {
    window.localStorage.removeItem(TOKEN_KEY);
  }

  /* Turn a FastAPI error response into one readable line. */
  function errorMessage(xhr, fallback) {
    var detail = xhr && xhr.responseJSON && xhr.responseJSON.detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail) && detail.length) {
      return detail
        .map(function (item) { return item.msg || JSON.stringify(item); })
        .join("; ");
    }
    if (xhr && xhr.status === 0) {
      return "Cannot reach the server.";
    }
    return fallback || "Request failed.";
  }

  /* api(method, url, ajaxOptions) -> jqXHR with the Bearer header attached. */
  function api(method, url, options) {
    var opts = $.extend({ method: method, url: url, dataType: "json", headers: {} }, options || {});
    var token = getToken();
    if (token) {
      opts.headers.Authorization = "Bearer " + token;
    }
    return $.ajax(opts);
  }

  api.json = function (method, url, body) {
    return api(method, url, { contentType: "application/json", data: JSON.stringify(body) });
  };

  api.form = function (method, url, body) {
    return api(method, url, { data: body });
  };

  function parseSseBlock(block, onEvent) {
    var payload = block
      .split("\n")
      .filter(function (line) { return line.indexOf("data:") === 0; })
      .map(function (line) { return line.slice(5).trim(); })
      .join("\n");
    if (!payload) {
      return;
    }
    try {
      onEvent(JSON.parse(payload));
    } catch (err) {
      /* Skip keepalives and partial frames. */
    }
  }

  /* POST JSON and invoke onEvent for each SSE `data:` object. */
  api.stream = function (url, body, onEvent) {
    var headers = { "Content-Type": "application/json", Accept: "text/event-stream" };
    var token = getToken();
    if (token) {
      headers.Authorization = "Bearer " + token;
    }
    return fetch(url, { method: "POST", headers: headers, body: JSON.stringify(body) })
      .then(function (res) {
        if (res.status === 401 || res.status === 403) {
          clearToken();
          window.location.replace("/login");
          return Promise.reject(res);
        }
        if (!res.ok) {
          return res.json().catch(function () { return {}; }).then(function (data) {
            var message = typeof data.detail === "string" ? data.detail : "Request failed.";
            var err = new Error(message);
            err.status = res.status;
            throw err;
          });
        }
        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";
        function pump() {
          return reader.read().then(function (chunk) {
            if (chunk.done) {
              if (buffer.trim()) {
                parseSseBlock(buffer, onEvent);
              }
              return;
            }
            buffer += decoder.decode(chunk.value, { stream: true });
            var parts = buffer.split("\n\n");
            buffer = parts.pop();
            parts.forEach(function (block) { parseSseBlock(block, onEvent); });
            return pump();
          });
        }
        return pump();
      });
  };

  window.App = {
    api: api,
    getToken: getToken,
    setToken: setToken,
    clearToken: clearToken,
    errorMessage: errorMessage
  };
})(window, jQuery);
