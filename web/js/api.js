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

  window.App = {
    api: api,
    getToken: getToken,
    setToken: setToken,
    clearToken: clearToken,
    errorMessage: errorMessage
  };
})(window, jQuery);
