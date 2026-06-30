"""
agent.py  —  DualLLMAgent

Main agent loop implementing the two-LLM architecture:

    Observe  →  Planner  →  Executor LLM  →  Execute  →  Observe  →  …

Planner  (LLM 1): Receives screenshot + OCR + DOM summary + goal + history.
                  Understands the UI, infers the workflow, returns a plan.

Executor LLM (LLM 2): Receives the plan + fresh screenshot.
                       Produces exactly ONE action dict — no long-term reasoning.

ActionExecutor (Selenium): Receives the action dict, performs the browser action.

The old VisionAgent class is preserved for backwards compatibility but the
active class is DualLLMAgent.  Both share all helper methods via the common
base-class pattern.
"""

import time
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# OCR HELPER  (used by both agents)
# ============================================================================

def _run_ocr_text(screenshot: Image.Image) -> str:
    """
    Run pytesseract on the screenshot and return a flat string of detected
    words ordered left-to-right, top-to-bottom.

    Returns "" if pytesseract is not installed or fails — the Planner
    degrades gracefully to vision-only mode in that case.
    """
    try:
        import pytesseract
    except ImportError:
        return ""

    try:
        tsv = pytesseract.image_to_data(
            screenshot,
            output_type=pytesseract.Output.DICT,
            config="--psm 11",
        )
        words = []
        for i, word in enumerate(tsv["text"]):
            word = word.strip()
            conf = int(tsv["conf"][i])
            if word and conf >= 30:
                words.append(word)
        return " ".join(words)
    except Exception as e:
        logger.debug(f"[OCR] pytesseract failed: {e}")
        return ""


# ============================================================================
# DUAL-LLM AGENT
# ============================================================================

class DualLLMAgent:
    """
    Two-LLM browser agent.

    Constructor parameters
    ----------------------
    browser        : BrowserController
    planner        : GeminiPlanner            (LLM 1 — UI reasoning + plan)
    executor_llm   : GeminiExecutorLLM        (LLM 2 — action translation)
    executor       : ActionExecutor           (Selenium action runner)
    ocr_provider   : GeminiVisionModel | None (used ONLY for OCR bounding boxes
                                               so executor.py can resolve click
                                               coordinates; no analyze() called)
    max_steps      : int
    memory_size    : int
    screenshot_dir : str | None
    """

    def __init__(
        self,
        browser,
        planner,
        executor_llm,
        executor,
        ocr_provider    = None,
        max_steps:      int = 40,
        memory_size:    int = 10,
        screenshot_dir: Optional[str] = "screenshots",
    ):
        self.browser        = browser
        self.planner        = planner
        self.executor_llm   = executor_llm
        self.executor       = executor
        self.ocr_provider   = ocr_provider     # for coordinate resolution only
        self.max_steps      = max_steps
        self.memory_size    = memory_size
        self.screenshot_dir = screenshot_dir
        self.action_history = []
        self.current_step   = 0

        if screenshot_dir:
            Path(screenshot_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # MAIN LOOP:  Observe → Planner → Executor LLM → Execute → Observe
    # -------------------------------------------------------------------------

    def run(self, start_url: str, goal: str) -> dict:
        logger.info("=" * 60)
        logger.info("DUAL-LLM AGENT STARTING")
        logger.info(f"Goal: {goal.strip()[:120]}")
        logger.info(f"URL : {start_url}")
        logger.info(f"Max steps: {self.max_steps}")
        logger.info("=" * 60)

        self.browser.navigate(start_url)
        time.sleep(2)

        while self.current_step < self.max_steps:
            self.current_step += 1
            logger.info(f"\n{'─'*40}")
            logger.info(f"STEP {self.current_step} / {self.max_steps}")
            logger.info(f"{'─'*40}")

            # ── OBSERVE ───────────────────────────────────────────────────────
            screenshot = self._observe()

            # ── PLANNER (LLM 1) ───────────────────────────────────────────────
            plan = self._plan_with_retry(screenshot, goal)
            logger.info(
                f"[PLAN] page={plan.get('page_context','?')} | "
                f"step='{plan.get('workflow_step','?')[:50]}'"
            )

            # Terminal check from Planner
            if plan.get("status") == "done":
                logger.info("PLANNER reports GOAL ACHIEVED.")
                return self._build_result("done", "Task completed successfully.")
            if plan.get("status") == "failed":
                logger.warning("PLANNER reports failure.")
                return self._build_result("failed", plan.get("reasoning", "Planner reported failure."))

            # ── EXECUTOR LLM (LLM 2) ─────────────────────────────────────────
            action_dict = self._execute_llm_with_retry(plan, screenshot)
            logger.info(
                f"[EXEC-LLM] [{action_dict.get('action','?').upper()}] "
                f"'{action_dict.get('target_description','')[:60]}'"
            )
            logger.info(f"[EXEC-LLM] {action_dict.get('reasoning','')[:100]}")

            # Terminal check from Executor LLM
            status = action_dict.get("status", "acting")
            action = action_dict.get("action", "")
            if status == "done" or action == "done":
                logger.info("EXECUTOR LLM confirms GOAL ACHIEVED.")
                return self._build_result("done", "Task completed successfully.")
            if status == "failed":
                logger.warning("EXECUTOR LLM reports failure.")
                return self._build_result("failed", action_dict.get("reasoning", ""))

            # ── EXECUTE (Selenium) ────────────────────────────────────────────
            executed = self._act_with_retry(action_dict, screenshot)
            self._update_memory(executed)

            if executed.get("result") == "task_complete":
                return self._build_result("done", "Task completed successfully.")

            # ── STUCK DETECTION ───────────────────────────────────────────────
            if self._is_stuck():
                stuck_target = self._get_stuck_scroll_target()
                if stuck_target == "sidebar":
                    logger.warning(
                        "Stuck sidebar scroll detected — escalating to "
                        "find_in_sidebar compound action."
                    )
                    sidebar_item = self._extract_sidebar_target_from_goal(goal)
                    recovery     = self.executor._find_in_sidebar(sidebar_item)
                    logger.info(f"find_in_sidebar recovery result: {recovery}")
                    self._update_memory({
                        "action":             "find_in_sidebar",
                        "target_description": sidebar_item,
                        "input_text":         "",
                        "result":             recovery,
                        "reasoning":          "stuck-loop recovery",
                    })
                else:
                    logger.warning("Stuck loop detected — scrolling to un-stick.")
                    self.executor._execute_scroll("down", "page")
                time.sleep(1)

            time.sleep(0.5)

        return self._build_result(
            "max_steps_reached",
            f"Task not completed within {self.max_steps} steps."
        )

    # -------------------------------------------------------------------------
    # OBSERVE
    # -------------------------------------------------------------------------

    def _observe(self) -> Image.Image:
        logger.info("[OBSERVE] Taking screenshot...")
        if self.screenshot_dir:
            ts        = datetime.now().strftime("%H%M%S")
            save_path = (
                f"{self.screenshot_dir}/step_{self.current_step:02d}_{ts}.png"
            )
            self.browser.take_screenshot(save_path=save_path)
        screenshot = self.browser.take_screenshot()
        logger.info(f"[OBSERVE] {screenshot.width}x{screenshot.height}px")
        return screenshot

    # -------------------------------------------------------------------------
    # PLANNER  (LLM 1)
    # -------------------------------------------------------------------------

    def _plan_with_retry(
        self, screenshot: Image.Image, goal: str, max_retries: int = 3
    ) -> dict:
        """
        Call the Planner with OCR text + DOM summary + screenshot.
        Retries up to max_retries times on failure.
        """
        from planner import extract_dom_summary, _fallback_plan

        recent_history = self.action_history[-self.memory_size:]

        # Gather supporting context (non-fatal if either fails)
        ocr_text = _run_ocr_text(screenshot)
        try:
            dom_summary = extract_dom_summary(self.browser.driver)
        except Exception as e:
            logger.debug(f"[PLAN] DOM summary failed (non-fatal): {e}")
            dom_summary = ""

        logger.info(
            f"[PLAN] OCR={len(ocr_text.split())} words | "
            f"DOM={len(dom_summary)} chars"
        )

        for attempt in range(max_retries):
            try:
                plan = self.planner.plan(
                    screenshot     = screenshot,
                    goal           = goal,
                    action_history = recent_history,
                    ocr_text       = ocr_text,
                    dom_summary    = dom_summary,
                )
                if plan and plan.get("next_action", {}).get("action"):
                    return plan
                logger.warning(f"[PLAN] Attempt {attempt+1}: invalid plan returned")
            except Exception as e:
                logger.warning(f"[PLAN] Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        logger.error("[PLAN] All Planner attempts failed — using scroll fallback plan.")
        return _fallback_plan("all retries exhausted")

    # -------------------------------------------------------------------------
    # EXECUTOR LLM  (LLM 2)
    # -------------------------------------------------------------------------

    def _execute_llm_with_retry(
        self, plan: dict, screenshot: Image.Image, max_retries: int = 2
    ) -> dict:
        from executor_llm import _fallback_action

        na = plan.get("next_action", {})

        # SHORT-CIRCUIT: compound actions that the Executor LLM must not override.
        # These actions are designed to handle situations where the target is NOT
        # yet visible on screen — so asking the Executor LLM to confirm would
        # always produce a wrong "wait" or "scroll" override.
        if na.get("action") in ("find_in_sidebar",):
            logger.info(
                "[EXEC-LLM] Bypassing LLM for compound action "
                f"'{na['action']}' — passing Planner's plan directly."
            )
            return {
                "status":             plan.get("status", "acting"),
                "action":             na.get("action"),
                "target_description": na.get("target_description", ""),
                "input_text":         na.get("input_text", ""),
                "scroll_direction":   na.get("scroll_direction", "down"),
                "scroll_target":      na.get("scroll_target", "sidebar"),
                "reasoning":          "Compound action — bypassed Executor LLM.",
            }

        # Normal path for all other actions
        for attempt in range(max_retries):
            try:
                action_dict = self.executor_llm.decide(
                    plan       = plan,
                    screenshot = screenshot,
                )
                if action_dict and action_dict.get("action"):
                    return action_dict
                logger.warning(f"[EXEC-LLM] Attempt {attempt+1}: invalid action returned")
            except Exception as e:
                logger.warning(f"[EXEC-LLM] Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        logger.warning(
            "[EXEC-LLM] All attempts failed — using Planner's next_action directly."
        )
        return {
            "status":             plan.get("status", "acting"),
            "action":             na.get("action",             "scroll"),
            "target_description": na.get("target_description", "page"),
            "input_text":         na.get("input_text",         ""),
            "scroll_direction":   na.get("scroll_direction",   "down"),
            "scroll_target":      na.get("scroll_target",      "page"),
            "reasoning":          "Executor LLM unavailable — using Planner's plan directly.",
        }

    # -------------------------------------------------------------------------
    # ACT  (Selenium ActionExecutor)
    # -------------------------------------------------------------------------

    def _act_with_retry(
        self, action_dict: dict, screenshot: Image.Image, max_retries: int = 2
    ) -> dict:
        for attempt in range(max_retries):
            try:
                result     = self.executor.execute(action_dict, retry_count=attempt)
                result_str = result.get("result", "")
                if result_str.startswith("success") or result_str == "task_complete":
                    return result
                logger.info(f"[ACT] Attempt {attempt+1}: {result_str}")
            except Exception as e:
                logger.warning(f"[ACT] Attempt {attempt+1} raised: {e}")

            if attempt < max_retries - 1:
                scroll_target = action_dict.get("scroll_target", "page")
                if scroll_target != "sidebar":
                    logger.info("[ACT] Scrolling before retry...")
                    self.executor._execute_scroll("down", "page")
                else:
                    logger.info("[ACT] Sidebar action failed — re-parking mouse...")
                    try:
                        vh = self.executor.driver.execute_script(
                            "return window.innerHeight;"
                        )
                        self.executor._park_mouse_in_sidebar(40, vh // 2)
                    except Exception:
                        pass
                time.sleep(1)

        return action_dict

    # -------------------------------------------------------------------------
    # MEMORY + STUCK DETECTION  (unchanged from VisionAgent)
    # -------------------------------------------------------------------------

    def _update_memory(self, action: dict):
        self.action_history.append({
            "step":               self.current_step,
            "action":             action.get("action",             "?"),
            "target_description": action.get("target_description", "?"),
            "input_text":         action.get("input_text",         ""),
            "result":             action.get("result",             "?"),
            "reasoning":          action.get("reasoning",          ""),
            "scroll_target":      action.get("scroll_target",      "page"),
        })

    def _is_stuck(self) -> bool:
        if len(self.action_history) < 3:
            return False
        last3   = self.action_history[-3:]
        actions = [a["action"] for a in last3]
        targets = [a["target_description"][:30] for a in last3]
        stuck   = len(set(actions)) == 1 and len(set(targets)) == 1
        if not stuck:
            return False

        form_field_keywords = {
            "username", "password", "email", "user", "pass",
            "input", "field", "text", "login", "name",
        }
        target_lower = targets[0].lower()
        if actions[0] == "click" and any(k in target_lower for k in form_field_keywords):
            logger.debug(
                f"Repeated click on form field '{targets[0]}' — not treating as stuck."
            )
            return False

        logger.warning(f"Stuck loop: [{actions[0]}] '{targets[0]}' x3")
        return True

    def _get_stuck_scroll_target(self) -> str:
        if len(self.action_history) < 1:
            return "page"
        return self.action_history[-1].get("scroll_target", "page")

    def _extract_sidebar_target_from_goal(self, goal: str) -> str:
        patterns = [
            r"[Ff]ind ['\"]?([A-Za-z ]+?)['\"]? in the sidebar",
            r"[Cc]lick ['\"]?([A-Za-z ]+?)['\"]? in the sidebar",
            r"sidebar.*?['\"]([A-Za-z ]+?)['\"]",
            r"['\"]([A-Za-z ]+?)['\"].*?sidebar",
        ]
        for pat in patterns:
            m = re.search(pat, goal)
            if m:
                return m.group(1).strip()
        quoted = re.findall(r"['\"]([A-Za-z ]{3,30})['\"]", goal)
        for q in quoted:
            if q.lower() not in {"login", "sign in", "submit"}:
                return q
        return "Service Note"

    # -------------------------------------------------------------------------
    # RESULT + SUMMARY
    # -------------------------------------------------------------------------

    def _build_result(self, status: str, message: str) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"AGENT FINISHED - {status}")
        logger.info(f"Steps: {self.current_step}  |  {message}")
        if self.screenshot_dir:
            logger.info(f"Screenshots: {self.screenshot_dir}/")
        logger.info(f"{'='*60}\n")
        return {
            "status":         status,
            "steps_taken":    self.current_step,
            "action_history": self.action_history,
            "message":        message,
            "screenshot_dir": self.screenshot_dir,
        }

    def get_session_summary(self) -> str:
        lines = [f"Session Summary ({self.current_step} steps):", ""]
        for a in self.action_history:
            lines.append(
                f"  Step {a['step']:2d}: [{a['action'].upper():16s}] "
                f"{a['target_description'][:45]:45s} -> {a['result']}"
            )
        return "\n".join(lines)


# ============================================================================
# LEGACY ALIAS  (backwards compatibility — existing code that imports VisionAgent
#               will still work; it simply delegates to DualLLMAgent internals)
# ============================================================================

class VisionAgent(DualLLMAgent):
    """
    Backwards-compatible wrapper.

    The original VisionAgent accepted (browser, vision_model, executor).
    This subclass maps those old positional arguments onto DualLLMAgent.

    If you still have code that does:
        agent = VisionAgent(browser, vision_model, executor)
    it will continue to work — vision_model.analyze() is used as a
    single-LLM fallback (Planner = vision_model, Executor LLM = passthrough).
    """

    def __init__(
        self,
        browser,
        vision_model,
        executor,
        max_steps:      int = 25,
        memory_size:    int = 5,
        screenshot_dir: Optional[str] = "screenshots",
    ):
        # Wrap the old vision_model so it satisfies the Planner interface.
        # The _LegacyPlannerAdapter calls vision_model.analyze() and wraps
        # the result into an ExecutionPlan dict.
        planner      = _LegacyPlannerAdapter(vision_model)
        executor_llm = _PassthroughExecutorLLM()

        super().__init__(
            browser        = browser,
            planner        = planner,
            executor_llm   = executor_llm,
            executor       = executor,
            ocr_provider   = vision_model,
            max_steps      = max_steps,
            memory_size    = memory_size,
            screenshot_dir = screenshot_dir,
        )


class _LegacyPlannerAdapter:
    """Wraps GeminiVisionModel.analyze() into the Planner interface."""

    def __init__(self, vision_model):
        self._vm = vision_model

    def plan(self, screenshot, goal, action_history, ocr_text="", dom_summary="") -> dict:
        action_dict = self._vm.analyze(
            screenshot     = screenshot,
            goal           = goal,
            action_history = action_history,
        )
        # Wrap action_dict into ExecutionPlan shape
        return {
            "status":        action_dict.get("status", "acting"),
            "page_context":  "unknown",
            "workflow_step": "legacy single-LLM mode",
            "next_action":   action_dict,
            "reasoning":     action_dict.get("reasoning", ""),
        }


class _PassthroughExecutorLLM:
    """Executor LLM shim that simply passes the plan's next_action through."""

    def decide(self, plan: dict, screenshot) -> dict:
        na = plan.get("next_action", {})
        return {
            "status":             plan.get("status", "acting"),
            "action":             na.get("action",             "scroll"),
            "target_description": na.get("target_description", "page"),
            "input_text":         na.get("input_text",         ""),
            "scroll_direction":   na.get("scroll_direction",   "down"),
            "scroll_target":      na.get("scroll_target",      "page"),
            "reasoning":          na.get("reasoning",          ""),
        }