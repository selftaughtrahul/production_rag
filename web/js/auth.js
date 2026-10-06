/* Auth gate, login, register, logout. Page role comes from <body data-page>. */
(function (window, $, App) {
  "use strict";

  var LOGIN = "/login";
  var DASHBOARD = "/dashboard";

  function go(path) {
    window.location.replace(path);
  }

  /* Resolve the current user from the stored token; clears a rejected token. */
  function currentUser() {
    if (!App.getToken()) {
      return $.Deferred().reject().promise();
    }
    return App.api("GET", "/auth/me").fail(function (xhr) {
      if (xhr.status === 401 || xhr.status === 403) {
        App.clearToken();
      }
    });
  }

  function showError($box, message) {
    $box.text(message).toggleClass("is-visible", Boolean(message));
  }

  function reveal() {
    $("body").removeClass("is-loading");
  }

  function onAuthenticated(res) {
    App.setToken(res.access_token);
    go(DASHBOARD);
  }

  function bindTabs() {
    $(".tab").on("click", function () {
      var target = $(this).data("target");
      $(".tab").removeClass("is-active");
      $(this).addClass("is-active");
      $(".auth-form").hide();
      $(target).show();
      showError($(".alert"), "");
    });
  }

  function bindForm($form, submit) {
    $form.on("submit", function (event) {
      event.preventDefault();
      var $btn = $form.find("button[type=submit]").prop("disabled", true);
      var $alert = $form.find(".alert");
      showError($alert, "");
      submit($form)
        .done(onAuthenticated)
        .fail(function (xhr) {
          showError($alert, App.errorMessage(xhr, "Could not sign in."));
        })
        .always(function () {
          $btn.prop("disabled", false);
        });
    });
  }

  function login($form) {
    // OAuth2 form body: "username" carries the email.
    return App.api.form("POST", "/auth/login", {
      username: $form.find("[name=email]").val().trim(),
      password: $form.find("[name=password]").val()
    });
  }

  function register($form) {
    return App.api.json("POST", "/auth/register", {
      username: $form.find("[name=username]").val().trim(),
      email: $form.find("[name=email]").val().trim(),
      password: $form.find("[name=password]").val()
    });
  }

  function bindLogout() {
    $("#logout").on("click", function () {
      App.clearToken();
      go(LOGIN);
    });
  }

  var pages = {
    gate: function () {
      currentUser()
        .done(function () { go(DASHBOARD); })
        .fail(function () { go(LOGIN); });
    },
    login: function () {
      currentUser()
        .done(function () { go(DASHBOARD); })
        .fail(function () {
          bindTabs();
          bindForm($("#login-form"), login);
          bindForm($("#register-form"), register);
          reveal();
        });
    },
    dashboard: function () {
      currentUser()
        .done(function (user) {
          $("#username").text(user.username);
          bindLogout();
          reveal();
        })
        .fail(function () { go(LOGIN); });
    }
  };

  $(function () {
    var page = $("body").data("page");
    if (pages[page]) {
      pages[page]();
    }
  });
})(window, jQuery, window.App);
