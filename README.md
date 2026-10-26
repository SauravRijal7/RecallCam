# RecallCam

RecallCam is a visual memory assistant that uses a webcam to recognize objects, remember where they were last seen, and help answer questions about them.

The idea behind RecallCam is simple: computers can already recognize objects, but what if they could remember them too?

## Features

- **Object Detection:** Detects everyday objects using YOLO.
- **Visual Memory:** Stores information about where and when objects were last seen.
- **Object Re-identification:** Uses visual features to help recognize the same object after it moves.
- **Hand Tracking:** Uses MediaPipe to detect hands and record holding events.
- **Voice Interaction:** Allows users to interact with the system through voice.
- **Memory Queries:** Answers questions such as "Where did I leave my phone?" and "What was I holding 10 seconds ago?"
- **Local Database:** Uses SQLite to store object observations and memory events.
- **Efficient Processing:** Answers supported questions directly from local memory, reducing unnecessary cloud AI usage.

## Tech Stack

- Python
- YOLO
- OpenCV
- MediaPipe
- SQLite
- Speech Recognition
- Text-to-Speech

## How It Works

1. The webcam captures frames from the surroundings.
2. YOLO detects objects in the frames.
3. Visual features help match detected objects with previously stored objects.
4. RecallCam saves observations, timestamps, and location information in a local database.
5. MediaPipe helps track hand interactions and record holding events.
6. When a question is asked, the system searches its stored memory to provide an answer.

## Getting Started

### Requirements

- Python
- A webcam
- A microphone for voice interaction
- Required model files and Python dependencies

### Installation

Clone the repository:

```bash
git clone https://github.com/SauravRijal7/RecallCam.git
cd RecallCam
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure the required YOLO weights and MediaPipe model files are available in the locations specified by the project configuration.

### Run the Application

```bash
python main.py
```

## Challenges

One of the main challenges was recognizing the same physical object across different frames and positions. Changing lighting, camera angles, false detections, and limited hardware resources also made real-time processing difficult.

Another challenge was combining object detection, hand tracking, persistent storage, and voice interaction into one system while keeping it responsive.

## Future Plans

- Improve object re-identification accuracy.
- Make long-term memory more reliable.
- Improve natural-language question answering.
- Optimize performance on low-resource computers.
- Explore support for multiple cameras.

## Project Status

RecallCam is an experimental prototype. Detection and memory results may vary depending on lighting, occlusion, camera position, and object visibility.

## Author

**Saurav Rijal**

GitHub: [SauravRijal7](https://github.com/SauravRijal7)

---

RecallCam explores how computer vision can move beyond recognizing objects to remembering them over time.
