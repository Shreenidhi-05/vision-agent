"""
executor_llm.py  —  Executor LLM (LLM 2 of 2)

Responsibility:
  - Receives:  the Planner's ExecutionPlan + a fresh screenshot
  - Returns:   exactly ONE action dict ready to pass to ActionExecutor.execute()

The Executor LLM performs NO long-term reasoning.
Its only job is to translate the Planner's intent into a precise,
well-formed action dict that ActionExecutor can run without ambiguity.

It answers questions like:
  - "The plan says click 'Proceed' — is 'Proceed' still visible on screen?"
  - "The plan says type into 'Business Unit' — is that the right field label?"
  - "Are there any visible validation errors I should report?"

Action dict schema (returned):
{
  "status":             "acting" | "done" | "failed",
  "action":             "click|type|select|hover|find_in_sidebar|scroll|wait|done",
  "target_description": "<exact visible text label>",
  "input_text":         "<text to type / option to select, or ''>",
  "scroll_direction":   "down|up",
  "scroll_target":      "page|sidebar",
  "reasoning":          "<1 sentence confirmation or correction>",
}
"""

import io
import json
import logging
import re
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EXECUTOR SYSTEM PROMPT
# ---------------------------------------------------------------------------

_EXECUTOR_SYSTEM = """\
You are the EXECUTOR component of a two-LLM autonomous browser agent.

YOUR ROLE:
  You receive a structured plan from the Planner LLM. Your job is to:
  1. Confirm the planned action is correct given the current screenshot.
  2. Output exactly one action dict — nothing more.

YOU DO NOT perform long-term reasoning. You do not decide what the overall
workflow is. The Planner already did that. You only confirm and execute.

CORRECTION AUTHORITY:
  You MAY correct the Planner if you see an obvious discrepancy:
    - Target element not visible on screen → use "wait" or "scroll" instead.
    - Modal has closed unexpectedly → report "wait" so agent can re-observe.
    - Visible error message → use "wait" and note the error in reasoning.
  In all other cases, follow the plan exactly.

AVAILABLE ACTIONS:
  "click"           — click a button, link, or UI element
  "type"            — type text into the currently focused field
  "select"          — choose from a dropdown / combobox / MUI Select
  "hover"           — hover to expand a menu
  "find_in_sidebar" — compound: expand sidebar + scroll + click nav item
  "scroll"          — scroll page or sidebar
  "wait"            — wait 2 s for UI to settle
  "done"            — task confirmed complete (visible success message)

OUTPUT EXACTLY THIS JSON — no markdown, no extra keys, no explanation outside braces:

{
  "status":             "acting",
  "action":             "<action name>",
  "target_description": "<exact visible text label>",
  "input_text":         "",
  "scroll_direction":   "down",
  "scroll_target":      "page",
  "reasoning":          "<1 sentence>"
}

RULES:
1. target_description must be the exact visible text label on screen.
2. For "select": input_text = the option to pick, or "" to auto-pick first.
3. For "type": input_text = the text to type.
4. For "scroll": scroll_direction = "down" or "up"; scroll_target = "page" or "sidebar".
5. Never output more than one action.
6. If the planned action is clearly wrong, output "wait" and explain in reasoning.

CRITICAL OVERRIDE RULE:
If the planned action is "find_in_sidebar", you MUST output that exact action
unchanged — never replace it with "wait" or "scroll".
The "find_in_sidebar" action is a compound action that handles sidebar hover,
expansion, scrolling, and clicking internally. The target element will NOT be
visible on screen before this action runs — that is expected and correct.
Do not second-guess it based on what you can see.
"""


def _build_executor_prompt(plan: dict) -> str:
    """
    Build the Executor's user-turn prompt from the Planner's output.
    Keep it short — the Executor only needs the plan and the screenshot.
    """
    na = plan.get("next_action", {})
    return f"""PLANNER'S PLAN:
  page_context  : {plan.get('page_context', 'unknown')}
  workflow_step : {plan.get('workflow_step', 'unknown')}
  planned action: {na.get('action', 'scroll')}
  target        : {na.get('target_description', 'page')}
  input_text    : {na.get('input_text', '')}
  scroll_dir    : {na.get('scroll_direction', 'down')}
  scroll_target : {na.get('scroll_target', 'page')}
  planner notes : {plan.get('reasoning', '')}

Look at the screenshot.
Confirm the planned action is correct, or make a minor correction if needed.
Output exactly one action dict as JSON.
"""


# ---------------------------------------------------------------------------
# GEMINI EXECUTOR LLM
# ---------------------------------------------------------------------------

class GeminiExecutorLLM:
    """
    Executor LLM backed by Google Gemini.

    Temperature is intentionally near-zero (default 0.1) for deterministic,
    precise action output.  The Executor should be a reliable transcription
    of the Planner's intent, not an independent reasoner.

    Interface:
        action_dict = executor_llm.decide(
            plan       = ExecutionPlan dict from GeminiPlanner,
            screenshot = PIL.Image,
        )
        # Returns: action dict ready for ActionExecutor.execute()
    """

    def __init__(
        self,
        api_key:     str,
        model_name:  str   = "gemini-2.5-flash",
        temperature: float = 0.1,
    ):
        self.api_key     = api_key
        self.model_name  = model_name
        self.temperature = temperature
        self._client     = None
        self._setup()

    def _setup(self):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Run: pip install google-genai")
        try:
            self._client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"GeminiExecutorLLM init failed: {e}")
        logger.info(f"GeminiExecutorLLM ready: {self.model_name}  temp={self.temperature}")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def decide(
        self,
        plan:       dict,
        screenshot: Image.Image,
    ) -> dict:
        """
        Translate the Planner's ExecutionPlan into a single action dict.

        Parameters
        ----------
        plan       : ExecutionPlan dict from GeminiPlanner.plan()
        screenshot : Fresh PIL screenshot (taken after Planner finished)

        Returns
        -------
        dict  action dict for ActionExecutor.execute()
        """
        img_bytes    = self._pil_to_jpeg_bytes(screenshot)
        user_prompt  = _build_executor_prompt(plan)
        raw          = self._call_gemini(img_bytes, user_prompt)
        action_dict  = _parse_action(raw)

        # If the Executor outputs "done" or "failed", propagate the plan
        # status so the agent loop can terminate correctly.
        if plan.get("status") in ("done", "failed") and action_dict.get("action") not in ("done",):
            # Trust the Planner's terminal status; override executor action.
            action_dict["status"] = plan["status"]

        logger.info(
            f"[EXECUTOR-LLM] [{action_dict.get('action','?').upper()}] "
            f"'{action_dict.get('target_description','')[:60]}'"
        )
        if action_dict.get("reasoning"):
            logger.info(f"[EXECUTOR-LLM] {action_dict['reasoning'][:100]}")

        return action_dict

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _pil_to_jpeg_bytes(self, image: Image.Image) -> bytes:
        max_width = 1280
        if image.width > max_width:
            ratio  = max_width / image.width
            image  = image.resize((max_width, int(image.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _call_gemini(self, img_bytes: bytes, user_prompt: str) -> str:
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model    = self.model_name,
                contents = [
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    user_prompt,
                ],
                config = types.GenerateContentConfig(
                    system_instruction = _EXECUTOR_SYSTEM,
                    temperature        = self.temperature,
                    max_output_tokens  = 4000,    # Executor output is always small
                    response_mime_type = "application/json",
                ),
            )
            raw = response.text or ""
            logger.debug(f"[EXECUTOR-LLM] raw: {raw[:300]}")
            return raw

        except Exception as e:
            logger.error(f"[EXECUTOR-LLM] Gemini call failed: {e}")
            if "mime" in str(e).lower() or "application/json" in str(e).lower():
                return self._call_gemini_text_plain(img_bytes, user_prompt)
            return ""

    def _call_gemini_text_plain(self, img_bytes: bytes, user_prompt: str) -> str:
        """Fallback for older model versions that reject application/json MIME."""
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model    = self.model_name,
                contents = [
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    user_prompt,
                ],
                config = types.GenerateContentConfig(
                    system_instruction = _EXECUTOR_SYSTEM,
                    temperature        = self.temperature,
                    max_output_tokens  = 512,
                    response_mime_type = "text/plain",
                ),
            )
            return response.text or ""
        except Exception as e:
            logger.error(f"[EXECUTOR-LLM] text/plain fallback also failed: {e}")
            return ""


# ---------------------------------------------------------------------------
# ACTION DICT PARSER
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> dict:
    """
    Extract and validate the action dict JSON from the Executor's raw output.
    Uses the same robust repair strategy as vision_model._parse_response.
    """
    if not raw or not raw.strip():
        logger.warning("[EXECUTOR-LLM] Empty response — defaulting to wait.")
        return _fallback_action("empty executor response")

    cleaned = raw
    cleaned = re.sub(r"<think>.*?</think>",         "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?",               "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("`").strip()

    start = cleaned.find("{")
    if start == -1:
        logger.warning(f"[EXECUTOR-LLM] No JSON in response: {raw[:200]}")
        return _fallback_action("no JSON found")

    json_candidate = cleaned[start:]

    # Happy path
    last_brace = json_candidate.rfind("}")
    if last_brace != -1:
        try:
            parsed = json.loads(json_candidate[:last_brace + 1])
            return _fill_action_defaults(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Truncation repair
    pos = len(json_candidate) - 1
    while pos >= 0:
        pos = json_candidate.rfind("}", 0, pos)
        if pos == -1:
            break
        try:
            parsed = json.loads(json_candidate[:pos + 1])
            return _fill_action_defaults(parsed)
        except (json.JSONDecodeError, ValueError):
            pos -= 1

    logger.warning(f"[EXECUTOR-LLM] All parse attempts failed. Raw: {raw[:200]}")
    return _fallback_action("parse error")


def _fill_action_defaults(parsed: dict) -> dict:
    """Ensure all keys expected by ActionExecutor.execute() are present."""
    parsed.setdefault("status",             "acting")
    parsed.setdefault("action",             "wait")
    parsed.setdefault("target_description", "page")
    parsed.setdefault("input_text",         "")
    parsed.setdefault("scroll_direction",   "down")
    parsed.setdefault("scroll_target",      "page")
    parsed.setdefault("reasoning",          "")
    return parsed


def _fallback_action(reason: str) -> dict:
    """Safe default: wait 2 s and let the agent re-observe."""
    return {
        "status":             "acting",
        "action":             "wait",
        "target_description": "page",
        "input_text":         "",
        "scroll_direction":   "down",
        "scroll_target":      "page",
        "reasoning":          f"Executor fallback ({reason}): waiting to re-observe.",
    }