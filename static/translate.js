let streamingInProgress = false;
let updateQueue = "";

if (!language) {
  throw new Error("Language not specified.");
}
const translationBox = document.getElementById("translationBox");

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
/*
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

      //   if (wasAtBottom) {
      //       smoothScrollTo(translationBox, translationBox.scrollHeight, 800); // 800ms duration
      // }
    }
  });
}
*/

// On load, fetch full buffer
function loadFullContent() {
  fetch(`/translations/${language}?full=true`)
    .then((response) => response.json())
    .then((data) => {
      translationBox.innerHTML =
        data.text + '<span class="buffer-text">' + data.buffer + "</span>";
    });
}
// also load full content
function pollUpdates() {
  fetch(`/translations/${language}`)
    .then((response) => response.json())
    .then((data) => {
      const wasAtBottom = isScrolledToBottom();

      // Remove the last buffer-text span if it exists.
      const lastBufferText = translationBox.querySelector(".buffer-text");
      if (lastBufferText) {
        lastBufferText.remove();
      }

      // Remove the new-update class from any previous update elements as soon as a new update arrives.
      if (data.text.length > 0) {
        const previousUpdates = translationBox.querySelectorAll(".new-update");
        previousUpdates.forEach((elem) => {
          elem.classList.remove("new-update");
        });
      }

      // append new complete text
      const confirmedElement = document.createElement("span");
      confirmedElement.classList.add("new-update");
      confirmedElement.innerHTML = " " + data.text;
      translationBox.appendChild(confirmedElement);

      // add buffer element
      const bufferElement = document.createElement("span");
      bufferElement.classList.add("buffer-text");
      bufferElement.innerHTML = " " + data.buffer + "<br>".repeat(6);
      translationBox.appendChild(bufferElement);

      // scroll to bottom if at bottom
      if (wasAtBottom) {
        smoothScrollTo(translationBox, translationBox.scrollHeight, 800); // 800ms duration
      }
    })
    .catch((err) => console.log("Update error:", err));
}

/*

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
*/
