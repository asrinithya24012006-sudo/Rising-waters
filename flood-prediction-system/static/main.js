// Rising Waters — front-end interactivity
// 1. Simple form validation on the assessment form
// 2. Animate the risk gauge fill on the result pages

document.addEventListener("DOMContentLoaded", function () {
  // --- Form validation (index.html) ---
  const form = document.getElementById("predictForm");
  if (form) {
    form.addEventListener("submit", function (e) {
      let valid = true;
      const inputs = form.querySelectorAll("input[required]");
      inputs.forEach(function (input) {
        input.classList.remove("invalid");
        if (input.value.trim() === "" || isNaN(parseFloat(input.value))) {
          input.classList.add("invalid");
          valid = false;
        }
      });
      if (!valid) {
        e.preventDefault();
      }
    });

    form.querySelectorAll("input").forEach(function (input) {
      input.addEventListener("input", function () {
        input.classList.remove("invalid");
      });
    });
  }

  // --- Gauge fill animation (chance.html / no_chance.html) ---
  const fill = document.getElementById("resultFill");
  if (fill) {
    const target = parseFloat(fill.dataset.target) || 0;
    requestAnimationFrame(function () {
      setTimeout(function () {
        fill.style.height = Math.min(Math.max(target, 3), 100) + "%";
      }, 150);
    });
  }
});
