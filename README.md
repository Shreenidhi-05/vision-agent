# Vision-Based Autonomous Web Agent

A Python-based autonomous web automation agent that uses Large Language Models (LLMs), Selenium, OCR, and computer vision to understand webpages and perform tasks through natural language instructions.

Instead of relying on hardcoded selectors or predefined workflows, the agent observes the webpage, reasons about the current state, plans the next action, and executes browser interactions dynamically.

---

## Features

- Natural language task execution
- Vision-based webpage understanding
- OCR-powered text detection using Tesseract
- Selenium browser automation
- LLM-driven planning and decision making
- Dynamic navigation without hardcoded workflows
- Support for:
  - Clicking elements
  - Typing into forms
  - Hover interactions
  - Sidebar navigation
  - Scrolling within pages or containers
  - Waiting for asynchronous page loads

---

## Architecture

```
User Goal
    │
    ▼
Planner (Gemini)
    │
    ▼
Action Executor
    │
    ├── Selenium DOM interactions
    ├── OCR (Tesseract)
    ├── Vision-based element detection
    └── Browser Automation
```

The system follows a perception → planning → execution loop until the task is completed.

---

## Tech Stack

- Python
- Selenium
- Google Gemini API
- Tesseract OCR
- OpenCV
- Pillow

---

## Project Structure

```
vision_agent/
│
├── agent.py
├── browser.py
├── executor.py
├── planner.py
├── executor_llm.py
├── vision_model.py
├── prompts.py
├── config.py
├── main.py
│
├── screenshots/
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/vision-agent.git
cd vision-agent
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configure

Create a `.env` file and add your Gemini API key.

```env
GEMINI_API_KEY=your_api_key
```

Install Tesseract OCR and ensure it is available in your system PATH.

---

## Running

```bash
python main.py
```

Provide:

- Website URL
- Natural language goal

Example:

```
Log into the application using the provided credentials, navigate to Service Note, and create a new Service Note.
```

---

## How It Works

1. Opens the webpage.
2. Captures a screenshot.
3. Extracts OCR text.
4. Collects lightweight DOM information.
5. Sends the webpage context to the LLM.
6. Receives the next action.
7. Executes the action using Selenium.
8. Repeats until the task is completed.

---

## Current Capabilities

- Login automation
- Sidebar navigation
- Dynamic button detection
- OCR-based fallback clicking
- Hover-based menus
- Page and container scrolling
- Form filling
- Autonomous task execution

---

## Future Improvements

- Better DOM grounding
- Visual object detection
- Playwright support
- Memory across tasks
- Multi-agent planning
- Self-correction and recovery
- Improved shadow DOM and iframe support

---

## License

This project is intended for educational and research purposes.
