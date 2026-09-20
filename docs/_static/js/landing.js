// Tabs of the landing page hero panel.
// The panel goes through its tabs once on its own, then rests on the first one. It
// waits while the reader hovers or focuses it or while it is off screen, stops as soon
// as the reader picks a tab, and can be paused and replayed with the button under the
// plot. It never moves on its own for readers who prefer reduced motion.
(function () {
  const panel = document.querySelector(".aeon-hero-panel");
  if (!panel) {
    return;
  }
  const tabs = Array.from(panel.querySelectorAll(".aeon-hero-tab"));
  const view = panel.querySelector(".aeon-hero-view");
  const pause = panel.querySelector(".aeon-hero-pause");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const INTERVAL = 7000;
  let timer = null;
  let stopped = false;
  let visible = true;
  let engaged = false;

  function select(tab) {
    tabs.forEach(function (other) {
      const active = other === tab;
      other.setAttribute("aria-selected", active ? "true" : "false");
      other.tabIndex = active ? 0 : -1;
    });
    panel.dataset.lens = tab.dataset.lens;
    view.setAttribute("aria-labelledby", tab.id);
  }

  function current() {
    return tabs.findIndex(function (tab) {
      return tab.getAttribute("aria-selected") === "true";
    });
  }

  function advance() {
    const next = (current() + 1) % tabs.length;
    select(tabs[next]);
    if (next === 0) {
      // back to the start, one tour of the tabs is enough
      stopped = true;
      update();
    }
  }

  function update() {
    const play = !stopped && visible && !engaged && !reducedMotion.matches;
    if (play && timer === null) {
      timer = window.setInterval(advance, INTERVAL);
    } else if (!play && timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
    panel.classList.toggle("is-playing", play);
    if (pause) {
      // the button reflects what the reader asked for, not the short waits on hover
      const label = stopped ? "Play" : "Pause";
      pause.querySelector("span").textContent = label;
      pause.setAttribute(
        "aria-label",
        stopped ? "Play the demonstration" : "Pause the demonstration"
      );
    }
  }

  function pick(tab, focus) {
    stopped = true;
    select(tab);
    if (focus) {
      tab.focus();
    }
    update();
  }

  tabs.forEach(function (tab, index) {
    tab.addEventListener("click", function () {
      pick(tab, false);
    });
    tab.addEventListener("keydown", function (event) {
      const last = tabs.length - 1;
      const target = {
        ArrowRight: index === last ? 0 : index + 1,
        ArrowLeft: index === 0 ? last : index - 1,
        Home: 0,
        End: last,
      }[event.key];
      if (target === undefined) {
        return;
      }
      event.preventDefault();
      pick(tabs[target], true);
    });
  });

  if (pause) {
    pause.hidden = false;
    pause.addEventListener("click", function () {
      stopped = !stopped;
      // a pause on hover must not hold back a reader who just pressed play
      engaged = false;
      update();
    });
  }

  ["mouseenter", "focusin"].forEach(function (name) {
    panel.addEventListener(name, function () {
      engaged = true;
      update();
    });
  });
  ["mouseleave", "focusout"].forEach(function (name) {
    panel.addEventListener(name, function () {
      engaged = false;
      update();
    });
  });

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
      update();
    }).observe(panel);
  }
  if (reducedMotion.addEventListener) {
    reducedMotion.addEventListener("change", update);
  }

  select(tabs[Math.max(current(), 0)]);
  // the first overlay waits for the series to be drawn
  window.setTimeout(function () {
    panel.classList.add("is-ready");
    update();
  }, 1800);
})();
