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


const language = "{{ language }}";
const translationBox = document.getElementById("translationBox");
let streamingInProgress = false;
let updateQueue = "";

// On load, fetch full buffer
function loadFullContent() {
  fetch(`/translations/${language}?full=true`)
    .then((response) => response.json())
    .then((data) => {
      translationBox.innerHTML = data.text;
    });
}

// Custom smooth scroll function
function smoothScrollTo(element, target, duration) {
  const start = element.scrollTop;
  const change = target - start;
  const startTime = performance.now();

  function animateScroll(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Use easeInOutQuad easing for a smoother effect
    const easedProgress =
      progress < 0.5
        ? 2 * progress * progress
        : -1 + (4 - 2 * progress) * progress;
    element.scrollTop = start + change * easedProgress;
    if (elapsed < duration) {
      requestAnimationFrame(animateScroll);
    }
  }
  requestAnimationFrame(animateScroll);
}

// Helper to check if scrolled to bottom
function isScrolledToBottom() {
  return (
    translationBox.scrollHeight - translationBox.clientHeight <=
    translationBox.scrollTop + 1
  );
}

// Function to stream text letter-by-letter into an element.
function streamText(text, elem, callback) {
  let index = 0;
  function streamLetter() {
    if (index < text.length) {
      elem.innerHTML += text[index];
      index++;
      setTimeout(streamLetter, 50); // Adjust speed as needed
    } else if (callback) {
      callback();
    }
  }
  streamLetter();
}

// Function to process a new update letter-by-letter.
function processUpdate(newText) {
  streamingInProgress = true;

  // Remove the new-update class from any previous update elements as soon as a new update arrives.
  const previousUpdates = translationBox.querySelectorAll(".new-update");
  previousUpdates.forEach((elem) => {
    elem.classList.remove("new-update");
  });

  const wasAtBottom = isScrolledToBottom();
  // Create new element for latest update.
  const updateElem = document.createElement("span");
  updateElem.classList.add("new-update");
  translationBox.appendChild(updateElem);

  streamText(newText, updateElem, () => {
    // Do not remove "new-update" class here;
    // leave it until a subsequent update arrives.
    streamingInProgress = false;
    // If a new update was queued while streaming, process it next.
    if (updateQueue.length > 0) {
      const queuedText = updateQueue;
      updateQueue = "";
      processUpdate(queuedText);
      // I am not shure I should make autoscrolling
        
      //  if (wasAtBottom) {
        //    smoothScrollTo(translationBox, translationBox.scrollHeight, 800); // 800ms duration          
    //}
  });
}

// Poll new translation increments letter-by-letter.
function pollUpdates() {
  fetch(`/translations/${language}`)
    .then((response) => response.json())
    .then((data) => {
      if (data.text) {
        // If streaming is in progress, queue the new update.
        if (streamingInProgress) {
          updateQueue += data.text;
        } else {
          processUpdate(data.text);
        }
      }
    })
    .catch((err) => console.log("Update error:", err));
}

// Load full content once page loads.
loadFullContent();
// Start polling every second.
setInterval(pollUpdates, 1000);

