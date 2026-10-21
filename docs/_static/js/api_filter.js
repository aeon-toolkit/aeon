// Filter box of the API reference module pages.
// A search input is added above the first summary table of the page. Typing in it
// hides the rows whose name and description do not contain every typed word, then the
// tables and sections left without any row. Short pages do not get a filter box.
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
  // what gets hidden when all of its rows are: the wrapper the theme puts around each
  // table, and the sections below the page title
  const groups = Array.from(
    article.querySelectorAll(".pst-scrollable-table-container, section section"),
  ).filter(function (group) {
    return group.querySelector("table.autosummary");
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

  function apply() {
    const words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    rows.forEach(function (row, i) {
      const match = words.every(function (word) {
        return texts[i].includes(word);
      });
      row.hidden = !match;
      shown += match ? 1 : 0;
    });
    groups.forEach(function (group) {
      group.hidden = !group.querySelector("table.autosummary tbody tr:not([hidden])");
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
