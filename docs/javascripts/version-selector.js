/*
 * Version selector for the Material header on Read the Docs.
 *
 * Read the Docs serves every release of the docs under /<language>/<version>/
 * and exposes the list of active versions through its addons API (the same API
 * its floating flyout uses). This script reads that list and adds a <select> to
 * the header; choosing a version opens the same page in that version, or that
 * version's home page when the page does not exist there.
 *
 * It does nothing when the docs are not served from a Read the Docs style URL
 * (for example `mkdocs serve`), or when the API cannot be reached.
 */
(function () {
  "use strict";

  // /en/latest/user/install/  ->  ["en", "latest", "user/install/"]
  function parsePath(pathname) {
    var m = pathname.match(/^\/([a-z]{2}(?:[-_][A-Za-z]{2})?)\/([^/]+)\/(.*)$/);
    return m ? { language: m[1], version: m[2], rest: m[3] } : null;
  }

  // "latest" and "stable" first, then release tags from newest to oldest.
  function sortVersions(versions) {
    function rank(slug) {
      if (slug === "latest") return [0, []];
      if (slug === "stable") return [1, []];
      var nums = (slug.match(/\d+/g) || []).map(Number);
      return [2, nums];
    }
    function cmp(a, b) {
      var ra = rank(a.slug), rb = rank(b.slug);
      if (ra[0] !== rb[0]) return ra[0] - rb[0];
      var n = Math.max(ra[1].length, rb[1].length);
      for (var i = 0; i < n; i++) {
        var d = (rb[1][i] || 0) - (ra[1][i] || 0); // larger number first
        if (d !== 0) return d;
      }
      return a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0;
    }
    return versions.slice().sort(cmp);
  }

  function withTrailingSlash(url) {
    return url.charAt(url.length - 1) === "/" ? url : url + "/";
  }

  // Same page in the chosen version, plus the section anchor if any.
  function targetUrl(version, rest, hash) {
    return withTrailingSlash(version.urls.documentation) + rest + (hash || "");
  }

  function buildSelect(versions, current) {
    var select = document.createElement("select");
    select.className = "pn-version-select";
    select.setAttribute("aria-label", "Documentation version");
    sortVersions(versions).forEach(function (v) {
      var option = document.createElement("option");
      option.value = v.slug;
      option.textContent = v.slug;
      option.selected = v.slug === current;
      select.appendChild(option);
    });
    return select;
  }

  function insertInHeader(select) {
    var inner = document.querySelector(".md-header__inner");
    if (!inner) return false;
    var anchor =
      inner.querySelector(".md-header__source") ||
      inner.querySelector(".md-search");
    if (anchor) inner.insertBefore(select, anchor);
    else inner.appendChild(select);
    return true;
  }

  function navigate(versions, slug, rest) {
    var version = versions.filter(function (v) { return v.slug === slug; })[0];
    if (!version) return;
    var page = targetUrl(version, rest, window.location.hash);
    var home = withTrailingSlash(version.urls.documentation);
    // Go to the same page if it exists in that version, otherwise its home page.
    fetch(page, { method: "HEAD" })
      .then(function (r) { window.location.href = r.ok ? page : home; })
      .catch(function () { window.location.href = home; });
  }

  function init() {
    var loc = parsePath(window.location.pathname);
    if (!loc) return;
    var slugMeta = document.querySelector('meta[name="readthedocs-project-slug"]');
    var project = slugMeta ? slugMeta.content : window.location.hostname.split(".")[0];
    var api =
      "/_/addons/?api-version=1&client-version=0.6.0" +
      "&project-slug=" + encodeURIComponent(project) +
      "&version-slug=" + encodeURIComponent(loc.version) +
      "&url=" + encodeURIComponent(window.location.href);

    fetch(api)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        var versions = data && data.versions && data.versions.active;
        if (!versions || versions.length < 2) return; // nothing to choose from
        var select = buildSelect(versions, loc.version);
        select.addEventListener("change", function () {
          navigate(versions, select.value, loc.rest);
        });
        insertInHeader(select);
      })
      .catch(function () { /* no selector outside Read the Docs */ });
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { parsePath: parsePath, sortVersions: sortVersions, targetUrl: targetUrl };
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
