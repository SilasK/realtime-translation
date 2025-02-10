// Get all "navbar-burger" elements
const $navbarBurgers = Array.prototype.slice.call(
  document.querySelectorAll(".navbar-burger"),
  0
);

// Add a click event on each of them
$navbarBurgers.forEach((el) => {
  el.addEventListener("click", () => {
    // Get the target from the "data-target" attribute
    const target = el.dataset.target;
    const $target = document.getElementById(target);

    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    el.classList.toggle("is-active");
    $target.classList.toggle("is-active");
  });
});

// Implement font size change functionality.
// Cycle through an array of font sizes for the reading text.
// Implement font size change functionality.
(function () {
  const readingText = document.getElementById("translationBox");
  const increaseBtn = document.getElementById("font-size-increase");
  const decreaseBtn = document.getElementById("font-size-decrease");
  // Set a minimum and maximum font size (in rem)
  const minFontSize = 1.0; // 1 rem minimum
  const maxFontSize = 2.5; // 2 rem maximum
  const step = 0.2; // each click changes size by 0.2 rem
  // Get initial font size. Fallback to 1.2rem if not set.
  let currentFontSize =
    parseFloat(window.getComputedStyle(readingText).fontSize) / 16 || 1.2;

  function updateFontSize(newSize) {
    currentFontSize = parseFloat(newSize);
    readingText.style.fontSize = newSize + "rem";
    console.log("Font size updated to", newSize);
  }

  increaseBtn.addEventListener("click", function () {
    console.log("currentFontSize", currentFontSize);

    if (currentFontSize + step <= maxFontSize) {
      updateFontSize((currentFontSize + step).toFixed(2));
    }
  });

  decreaseBtn.addEventListener("click", function () {
    console.log("currentFontSize", currentFontSize);

    if (currentFontSize - step >= minFontSize) {
      updateFontSize((currentFontSize - step).toFixed(2));
    }
  });
})();
