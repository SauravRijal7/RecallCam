# RecallCam

RecallCam is a visual memory assistant that uses a webcam to recognize objects, remember where they were last seen, and answer questions about them.

The idea is to go beyond object detection and build a system that can remember what it sees over time.

## What It Does

- Detects everyday objects using computer vision.
- Remembers where and when objects were last seen.
- Uses visual features to recognize objects after they move.
- Tracks hand interactions and records holding events.
- Supports voice interaction and spoken responses.
- Answers questions such as "Where did I leave my phone?" and "What was I holding 10 seconds ago?"
- Stores observations locally using SQLite.

## How It Works

RecallCam processes webcam frames, detects objects, and matches them with previously observed objects using visual features. It stores observations and timestamps in a local database.

When a user asks a question, the system searches its stored memory to find relevant information and provide an answer.

## Built With

- Python
- YOLO
- OpenCV
- MediaPipe
- SQLite
- Speech Recognition
- Text-to-Speech

## Running the Project

Run the main application from the project directory:

```bash
python main.py
```

The required model files and Python packages must be available for the application to work.

## Challenges

Recognizing the same object across different positions, handling changing lighting conditions, reducing false detections, and maintaining real-time performance were some of the main challenges.

Combining object detection, hand tracking, persistent memory, and voice interaction into one system also required careful integration.

## What's Next

- Improve object recognition and re-identification.
- Make long-term memory more reliable.
- Improve natural-language question answering.
- Optimize performance on lower-powered computers.
- Explore multi-camera support.

## Project Status

RecallCam is an experimental prototype, and its accuracy depends on factors such as lighting, visibility, and camera position.

## Author

**Saurav Rijal**

[GitHub](https://github.com/SauravRijal7)

---

RecallCam explores a simple idea: computers should not only be able to see the world, but also remember it.
