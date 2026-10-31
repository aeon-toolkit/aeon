// Filter box of the API reference module pages.
// A search input is added above the first summary table of the page. Typing in it
// hides the rows where some typed word is neither in the name and description nor in a
// heading above the row, then the tables, sections and "On this page" entries left
// without any row. Short pages do not get a filter box.
(function () {
  const MIN_ROWS = 10;
  const article = document.querySelector("article.bd-article");
  if (!article) {
    return;
  }
  const rows = Array.from(article.querySelectorAll("table.autosummary tbody tr"));
  if (rows.length < MIN_ROWS) {
    return;
  }
  const texts = rows.map(function (row) {
    return row.textContent.toLowerCase();
  });
  // the words of the headings above each row, so that a section can be found by its
  // title. "based" is left out: it is in most category headings ("Shapelet-based"), and
  // a search for the base classes would show all of these sections.
  const headingWords = rows.map(function (row) {
    let words = [];
    let section = row.closest("section section");
    while (section) {
      const heading = section.querySelector(":scope > :is(h2, h3, h4, h5, h6)");
      if (heading) {
        words = words.concat(heading.textContent.toLowerCase().split(/[^a-z0-9]+/));
      }
      section = section.parentElement.closest("section section");
    }
    return words.filter(function (word) {
      return word && word !== "based";
    });
  });
  // what gets hidden when all of its rows are: the wrapper the theme puts around each
  // table, and the sections below the page title
  const groups = Array.from(
    article.querySelectorAll(".pst-scrollable-table-container, section section"),
  ).filter(function (group) {
    return group.querySelector("table.autosummary");
  });
  // the "On this page" entries of the sections, hidden with them
  const toc = document.getElementById("pst-page-toc-nav");
  const tocEntries = groups.map(function (group) {
    const link =
      toc && group.id && toc.querySelector('a[href="#' + CSS.escape(group.id) + '"]');
    return link ? link.closest("li") : null;
  });

  const box = document.createElement("div");
  box.className = "aeon-api-filter";
  box.setAttribute("role", "search");
  const input = document.createElement("input");
  input.type = "search";
  input.placeholder = "Filter this page by name or description";
  input.setAttribute("aria-label", "Filter the functions and classes of this page");
  const count = document.createElement("span");
  count.className = "aeon-api-filter-count";
  count.setAttribute("aria-live", "polite");
  box.append(input, count);

  // above the first table, or above the section it is in so that the heading stays
  // with its table
  const page = article.querySelector("section");
  let anchor = article.querySelector("table.autosummary");
  while (anchor.parentElement !== page && anchor.parentElement !== article) {
    anchor = anchor.parentElement;
  }
  anchor.before(box);

  // a typed word is in the headings when each of its parts starts a heading word, so
  // that "warp" and "soft-dtw" find "Soft Dynamic Time Warping (soft-DTW)"
  function inHeadings(word, headings) {
    const parts = word.split(/[^a-z0-9]+/).filter(Boolean);
    return (
      parts.length > 0 &&
      parts.every(function (part) {
        return headings.some(function (heading) {
          return heading.startsWith(part);
        });
      })
    );
  }

  function apply() {
    const words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    rows.forEach(function (row, i) {
      const match = words.every(function (word) {
        return texts[i].includes(word) || inHeadings(word, headingWords[i]);
      });
      row.hidden = !match;
      shown += match ? 1 : 0;
    });
    groups.forEach(function (group, i) {
      group.hidden = !group.querySelector("table.autosummary tbody tr:not([hidden])");
      if (tocEntries[i]) {
        tocEntries[i].hidden = group.hidden;
      }
    });
    if (words.length === 0) {
      count.textContent = "";
    } else if (shown === 0) {
      count.textContent = "No match";
    } else {
      count.textContent = shown + " of " + rows.length + " shown";
    }
  }

  input.addEventListener("input", apply);
})();
