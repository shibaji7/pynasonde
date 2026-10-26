// Unit tests for docs/javascripts/version-selector.js (run with: node tests/js/version_selector.test.js)
const assert = require("assert");
const sel = require("../../docs/javascripts/version-selector.js");

// parsePath: /<language>/<version>/<page>
assert.deepStrictEqual(sel.parsePath("/en/latest/user/install/"), { language: "en", version: "latest", rest: "user/install/" });
assert.deepStrictEqual(sel.parsePath("/en/v1.3.0/"), { language: "en", version: "v1.3.0", rest: "" });
assert.strictEqual(sel.parsePath("/user/install/"), null); // local `mkdocs serve`
assert.strictEqual(sel.parsePath("/"), null);

// sortVersions: latest, stable, then newest tag first (numeric: 1.10 > 1.9)
const slugs = ["v1.0.0", "stable", "v2.0.0", "v1.10.0", "latest", "v1.9.0", "v1.2.1", "v1.2.0", "v1.3.0", "v1.2.2"].map((s) => ({ slug: s }));
assert.deepStrictEqual(
  sel.sortVersions(slugs).map((v) => v.slug),
  ["latest", "stable", "v2.0.0", "v1.10.0", "v1.9.0", "v1.3.0", "v1.2.2", "v1.2.1", "v1.2.0", "v1.0.0"]
);

// Shape of the Read the Docs addons response (only the part the script reads)
const rtd = {
  versions: {
    active: [
      { slug: "stable", urls: { documentation: "https://pynasonde.readthedocs.io/en/stable/" } },
      { slug: "latest", urls: { documentation: "https://pynasonde.readthedocs.io/en/latest/" } },
    ],
  },
};
assert.deepStrictEqual(sel.sortVersions(rtd.versions.active).map((v) => v.slug), ["latest", "stable"]);

// targetUrl keeps the page and the anchor, and tolerates a missing trailing slash
const v13 = { slug: "v1.3.0", urls: { documentation: "https://pynasonde.readthedocs.io/en/v1.3.0/" } };
assert.strictEqual(sel.targetUrl(v13, "user/install/", "#core"), "https://pynasonde.readthedocs.io/en/v1.3.0/user/install/#core");
assert.strictEqual(sel.targetUrl({ urls: { documentation: "https://x/en/v1/" } }, "", ""), "https://x/en/v1/");
assert.strictEqual(sel.targetUrl({ urls: { documentation: "https://x/en/v1" } }, "a/", ""), "https://x/en/v1/a/");

console.log("version-selector tests passed");
