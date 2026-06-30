import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

START_URL = os.getenv("START_URL")
LOGIN_USERNAME = os.getenv("LOGIN_USERNAME")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Goal is now deliberately brief.
#     The Planner LLM figures out the intermediate steps by reading the UI.
GOAL = "Create a Service Note"

# --- Model names ---
PLANNER_MODEL      = "gemini-2.5-flash"
EXECUTOR_LLM_MODEL = "gemini-2.5-flash"

# --- Temperature ---
PLANNER_TEMPERATURE      = 0.6   # moderate: allows creative recovery reasoning
EXECUTOR_LLM_TEMPERATURE = 0.1   # near-zero: deterministic, precise action output

# --- Browser ---
HEADLESS      = False
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 900

# --- Agent ---
MAX_STEPS      = 40    # two LLM calls per step; more steps needed than single-LLM
MEMORY_SIZE    = 10
SCREENSHOT_DIR = "screenshots"


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt  = "%H:%M:%S",
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("agent_run.log", mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error(
            "\nGEMINI_API_KEY is not set!\n"
            "  Get one free at: https://aistudio.google.com/apikey\n"
        )
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("DUAL-LLM VISION BROWSER AGENT")
    logger.info(f"  Planner      : {PLANNER_MODEL}  temp={PLANNER_TEMPERATURE}")
    logger.info(f"  Executor LLM : {EXECUTOR_LLM_MODEL}  temp={EXECUTOR_LLM_TEMPERATURE}")
    logger.info("=" * 70)
    logger.info(f"URL  : {START_URL}")
    logger.info(f"User : {LOGIN_USERNAME}")
    logger.info(f"Goal : {GOAL}")
    logger.info("=" * 70)

    # ── Browser ───────────────────────────────────────────────────────────────
    logger.info("\n[INIT] Starting browser...")
    from browser import BrowserController
    browser = BrowserController(
        headless      = HEADLESS,
        window_width  = WINDOW_WIDTH,
        window_height = WINDOW_HEIGHT,
    )

    # ── LLM 1: Planner ────────────────────────────────────────────────────────
    logger.info("[INIT] Initialising Planner (LLM 1)...")
    from planner import GeminiPlanner
    try:
        planner = GeminiPlanner(
            api_key     = GEMINI_API_KEY,
            model_name  = PLANNER_MODEL,
            temperature = PLANNER_TEMPERATURE,
        )
        logger.info("[INIT] Planner ready.")
    except Exception as e:
        logger.error(f"[INIT] Planner init failed: {e}")
        browser.quit()
        sys.exit(1)

    # ── LLM 2: Executor LLM ───────────────────────────────────────────────────
    logger.info("[INIT] Initialising ExecutorLLM (LLM 2)...")
    from executor_llm import GeminiExecutorLLM
    try:
        executor_llm = GeminiExecutorLLM(
            api_key     = GEMINI_API_KEY,
            model_name  = EXECUTOR_LLM_MODEL,
            temperature = EXECUTOR_LLM_TEMPERATURE,
        )
        logger.info("[INIT] ExecutorLLM ready.")
    except Exception as e:
        logger.error(f"[INIT] ExecutorLLM init failed: {e}")
        browser.quit()
        sys.exit(1)

    # ── OCR provider  (GeminiVisionModel used ONLY for OCR coordinate lookup) ──
    # The Planner and Executor LLM handle all reasoning.
    # GeminiVisionModel.get_last_ocr_results() is still called by executor.py
    # to resolve click coordinates from text bounding boxes.
    logger.info("[INIT] Setting up OCR provider...")
    from vision_model import GeminiVisionModel
    try:
        ocr_provider = GeminiVisionModel(
            api_key    = GEMINI_API_KEY,
            model_name = PLANNER_MODEL,
        )
        logger.info("[INIT] OCR provider ready.")
    except Exception as e:
        logger.warning(f"[INIT] OCR provider unavailable (non-fatal): {e}")
        ocr_provider = None

    # ── Selenium action executor ──────────────────────────────────────────────
    logger.info("[INIT] Setting up ActionExecutor...")
    from executor import ActionExecutor
    executor = ActionExecutor(browser)
    # Wire the OCR provider so executor.py can call vision_model.get_last_ocr_results()
    executor.vision_model = ocr_provider

    # ── Agent ─────────────────────────────────────────────────────────────────
    logger.info("[INIT] Creating DualLLMAgent...")
    from agent import DualLLMAgent
    agent = DualLLMAgent(
        browser        = browser,
        planner        = planner,
        executor_llm   = executor_llm,
        executor       = executor,
        ocr_provider   = ocr_provider,
        max_steps      = MAX_STEPS,
        memory_size    = MEMORY_SIZE,
        screenshot_dir = SCREENSHOT_DIR,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    logger.info("\n[RUN] Agent starting...\n")
    try:
        # Include credentials in the goal so the Planner can handle login
        full_goal = (
            f"Log in with username '{LOGIN_USERNAME}' and password '{LOGIN_PASSWORD}'. "
            f"Then: {GOAL}."
        )
        result = agent.run(start_url=START_URL, goal=full_goal)

        logger.info("\n" + "=" * 70)
        logger.info("RESULT")
        logger.info("=" * 70)
        logger.info(f"Status  : {result['status']}")
        logger.info(f"Steps   : {result['steps_taken']}")
        logger.info(f"Message : {result['message']}")
        if SCREENSHOT_DIR:
            logger.info(f"Screenshots : {SCREENSHOT_DIR}/")
        logger.info("\n" + agent.get_session_summary())
        return result

    except KeyboardInterrupt:
        logger.info("\n[STOPPED] Interrupted by user.")
        return {"status": "interrupted"}

    except Exception as e:
        logger.error(f"\n[ERROR] {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

    finally:
        logger.info("\n[CLEANUP] Closing browser...")
        browser.quit()


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("status") == "done" else 1)