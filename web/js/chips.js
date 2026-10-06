/* Tag chip input: type a label, Enter/comma adds a chip, Backspace/x removes. */
(function (window, $, App) {
  "use strict";

  var MAX_TAGS = 20;
  var MAX_LEN = 32;
  var RESERVED = ["user_id", "document_id", "filename", "chunk_id", "chunk_index"];

  /* Mirror of normalize_tags() on the server. */
  function normalizeTag(value) {
    var tag = String(value || "").trim().toLowerCase().replace(/\s+/g, "-");
    tag = tag.replace(/[^a-z0-9_-]+/g, "").replace(/^[-_]+|[-_]+$/g, "").slice(0, MAX_LEN);
    if (!tag || RESERVED.indexOf(tag) !== -1) {
      return "";
    }
    return tag;
  }

  function normalizeTags(list) {
    var out = [];
    (list || []).forEach(function (item) {
      var tag = normalizeTag(item);
      if (tag && out.indexOf(tag) === -1 && out.length < MAX_TAGS) {
        out.push(tag);
      }
    });
    return out;
  }

  /* Read-only chip list for rendering stored tags. */
  function renderTags(tags) {
    var $wrap = $('<span class="tag-list"></span>');
    (tags || []).forEach(function (tag) {
      $wrap.append($('<span class="tag"></span>').text(tag));
    });
    return $wrap;
  }

  /*
   * chips($root, { placeholder, suggestions, onChange })
   * $root is an empty container; returns { get, set, clear, suggest }.
   */
  function chips($root, options) {
    var opts = options || {};
    var tags = [];
    var listId = "chips-suggest-" + Math.random().toString(36).slice(2, 8);
    var $list = $('<span class="chips-list"></span>');
    var $input = $('<input type="text" class="chips-input" autocomplete="off">')
      .attr("placeholder", opts.placeholder || $root.data("placeholder") || "Add tag")
      .attr("list", listId);
    var $datalist = $("<datalist></datalist>").attr("id", listId);

    $root.addClass("chips").empty().append($list, $input, $datalist);

    function paint() {
      $list.empty();
      tags.forEach(function (tag) {
        $list.append(
          $('<span class="chip"></span>')
            .text(tag)
            .append($('<button type="button" class="chip-x" aria-label="Remove">×</button>').attr("data-tag", tag))
        );
      });
      $input.prop("disabled", tags.length >= MAX_TAGS);
      if (typeof opts.onChange === "function") {
        opts.onChange(tags.slice());
      }
    }

    function add(raw) {
      var tag = normalizeTag(raw);
      if (tag && tags.indexOf(tag) === -1 && tags.length < MAX_TAGS) {
        tags.push(tag);
        paint();
      }
      $input.val("");
    }

    function remove(tag) {
      tags = tags.filter(function (item) { return item !== tag; });
      paint();
    }

    $input.on("keydown", function (event) {
      if (event.key === "Enter" || event.key === ",") {
        event.preventDefault();
        event.stopPropagation();
        add($input.val());
      } else if (event.key === "Backspace" && !$input.val() && tags.length) {
        event.preventDefault();
        remove(tags[tags.length - 1]);
      }
    });

    /* Picking a datalist entry fires "input"/"change" without a key. */
    $input.on("change", function () {
      var value = $input.val();
      if (value && value.indexOf(",") === -1 && opts.suggestions && opts.suggestions.indexOf(value) !== -1) {
        add(value);
      }
    });

    $input.on("blur", function () {
      if ($input.val().trim()) {
        add($input.val());
      }
    });

    $root.on("click", function (event) {
      var $x = $(event.target).closest(".chip-x");
      if ($x.length) {
        event.preventDefault();
        remove($x.data("tag"));
        return;
      }
      $input.focus();
    });

    var api = {
      get: function () { return tags.slice(); },
      set: function (list) { tags = normalizeTags(list); paint(); },
      clear: function () { tags = []; $input.val(""); paint(); },
      suggest: function (list) {
        opts.suggestions = normalizeTags(list);
        $datalist.empty();
        opts.suggestions.forEach(function (tag) {
          $datalist.append($("<option></option>").attr("value", tag));
        });
      },
      input: $input
    };

    if (opts.suggestions) {
      api.suggest(opts.suggestions);
    }
    paint();
    return api;
  }

  var MAX_META = 20;
  var MAX_VALUE = 64;

  function normalizeMetaEntry(raw) {
    var text = String(raw || "").trim();
    var sep = text.indexOf(":") >= 0 ? ":" : (text.indexOf("=") >= 0 ? "=" : "");
    if (!sep) {
      return null;
    }
    var parts = text.split(sep);
    var key = normalizeTag(parts.shift());
    var value = parts.join(sep).trim().replace(/\s+/g, " ").slice(0, MAX_VALUE);
    if (!key || !value) {
      return null;
    }
    return { key: key, value: value };
  }

  function normalizeMeta(raw) {
    var out = {};
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      return out;
    }
    Object.keys(raw).forEach(function (key) {
      if (Object.keys(out).length >= MAX_META) {
        return;
      }
      var entry = normalizeMetaEntry(key + ":" + raw[key]);
      if (entry) {
        out[entry.key] = entry.value;
      }
    });
    return out;
  }

  function renderMeta(metadata) {
    var $wrap = $('<span class="tag-list"></span>');
    Object.keys(metadata || {}).forEach(function (key) {
      $wrap.append($('<span class="tag tag-meta"></span>').text(key + ": " + metadata[key]));
    });
    return $wrap;
  }

  /* metaChips($root) stores key:value chips. get() returns an object. */
  function metaChips($root, options) {
    var opts = options || {};
    var entries = {};
    var $list = $('<span class="chips-list"></span>');
    var $input = $('<input type="text" class="chips-input" autocomplete="off">')
      .attr("placeholder", opts.placeholder || $root.data("placeholder") || "key: value");

    $root.addClass("chips").empty().append($list, $input);

    function paint() {
      $list.empty();
      Object.keys(entries).forEach(function (key) {
        $list.append(
          $('<span class="chip"></span>')
            .text(key + ": " + entries[key])
            .append($('<button type="button" class="chip-x" aria-label="Remove">×</button>').attr("data-key", key))
        );
      });
      $input.prop("disabled", Object.keys(entries).length >= MAX_META);
      if (typeof opts.onChange === "function") {
        opts.onChange($.extend({}, entries));
      }
    }

    function add(raw) {
      var entry = normalizeMetaEntry(raw);
      if (entry && Object.keys(entries).length < MAX_META) {
        entries[entry.key] = entry.value;
        paint();
      }
      $input.val("");
    }

    $input.on("keydown", function (event) {
      if (event.key === "Enter" || event.key === ",") {
        event.preventDefault();
        event.stopPropagation();
        add($input.val());
      } else if (event.key === "Backspace" && !$input.val()) {
        var keys = Object.keys(entries);
        if (keys.length) {
          event.preventDefault();
          delete entries[keys[keys.length - 1]];
          paint();
        }
      }
    });

    $input.on("blur", function () {
      if ($input.val().trim()) {
        add($input.val());
      }
    });

    $root.on("click", function (event) {
      var $x = $(event.target).closest(".chip-x");
      if ($x.length) {
        event.preventDefault();
        delete entries[$x.attr("data-key")];
        paint();
        return;
      }
      $input.focus();
    });

    paint();
    return {
      get: function () { return $.extend({}, entries); },
      set: function (value) { entries = normalizeMeta(value); paint(); },
      clear: function () { entries = {}; $input.val(""); paint(); }
    };
  }

  App.chips = chips;
  App.metaChips = metaChips;
  App.normalizeTags = normalizeTags;
  App.renderTags = renderTags;
  App.renderMeta = renderMeta;
})(window, jQuery, window.App);
