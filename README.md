# HandyAI

A web app that recognizes ASL fingerspelling in real time using your webcam and provides live text output and speech. Designed for accessibility, learning, and quick demos — no heavy installs required.

**Features**
- **Real-time Recognition:** Live webcam capture with immediate letter detection.
- **Hybrid Classification:** Rule-based heuristics, nearest-neighbor templates, and optional TensorFlow.js model support.
- **Personal Templates:** Save and load custom letter templates per user (localStorage).
- **Seeded Demo Templates:** Built-in seeded templates for instant demo without training.
- **Train & Persist:** Optional model training and saving to IndexedDB (`localstorage://asl-model`).
- **Download / Import:** Export and import template JSON for sharing or backup.
- **Visual Feedback:** Overlay canvas and visual state for hand tracking.

**Quick Start**
- Open the project folder and run a static server or open the files in a browser.
- For best results, run a local server so model and resources load properly.

Example (using Python 3):

```bash
python -m http.server 8000
# then open http://localhost:8000 in your browser
```

**Usage**
- Allow the browser to access your webcam when prompted.
- Show letters with one hand to the camera; the app will display and optionally speak the detected letter.
- Press keys `A`–`Z` while showing the corresponding sign to add samples to your account templates (login required for persistence).
- Use the UI buttons to download/import template sets or clear saved templates.

**Detected / Used Libraries & Frameworks**
- **MediaPipe Hands**: used for hand landmark detection (CDN @mediapipe/hands).
- **TensorFlow.js**: optional model training and inference via `tf.loadLayersModel` and `tf.save` APIs.
- **Web APIs**: `getUserMedia`, `Canvas`, `localStorage`, and IndexedDB for persistence.
- **No build tools required**: The project runs with plain HTML/CSS/JavaScript.

**Files**
- **Project entry:** [index.html](index.html)
- **Main logic:** [app.js](app.js)
- **Styles:** [style.css](style.css)

**Tips & Troubleshooting**
- If the optional TF model is not present, the app falls back to rule-based classification and template NN.
- For model training/saving, use a modern browser with IndexedDB support (Chrome/Edge/Firefox).
- Good lighting and a plain background improve detection stability.

**Contributing**
- Suggestions, bug reports, and PRs are welcome. Start by opening an issue describing the change.

**Credits**
- Hand detection powered by MediaPipe; ML support via TensorFlow.js.
