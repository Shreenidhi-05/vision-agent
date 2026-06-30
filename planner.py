"""
planner.py  —  Planner LLM (LLM 1 of 2)

Responsibility:
  - Receives:  screenshot (pixels), OCR text, DOM summary, goal, action history
  - Returns:   a structured ExecutionPlan dict describing the NEXT logical step
                and the OVERALL workflow the Planner infers from the UI state.

The Planner performs ALL UI reasoning:
  - What page / modal is currently visible?
  - What is the correct next workflow step?
  - What element should be acted on?
  - Why?

The Planner does NOT issue raw browser commands.
That responsibility belongs entirely to the Executor LLM (executor_llm.py).

ExecutionPlan schema (returned as dict):
{
  "status":       "acting" | "done" | "failed",
  "page_context": "login" | "post_login_nav" | "service_note_list" |
                  "create_modal" | "form_filling" | "unknown",
  "workflow_step": "<short human-readable description of where we are>",
  "next_action": {
      "action":             "click|type|select|hover|find_in_sidebar|scroll|wait|done",
      "target_description": "<exact visible text label or element description>",
      "input_text":         "<text to type / option to select, or empty string>",
      "scroll_direction":   "down|up",
      "scroll_target":      "page|sidebar",
  },
  "reasoning": "<1–3 sentences explaining why this action is the right next step>",
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
# DOM SUMMARY HELPER
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int = 120) -> str:
    return text if len(text) <= n else text[:n] + "…"


def _format_history(history: list, max_entries: int = 8) -> str:
    """
    Format the raw action history into a compact, readable string for the
    Planner prompt.  We show more history than the old single-LLM approach
    (8 vs 5 entries) because the Planner needs to infer the workflow stage.
    """
    if not history:
        return "  (none — this is the first step)"
    recent = history[-max_entries:]
    lines  = []
    for i, a in enumerate(recent):
        if not isinstance(a, dict):
            lines.append(f"  {i+1}. {a}")
            continue
        typed = f" | typed: '{_truncate(a['input_text'], 40)}'" if a.get("input_text") else ""
        lines.append(
            f"  {i+1}. [{a.get('action','?').upper():16s}] "
            f"'{_truncate(a.get('target_description','?'), 50)}'"
            f"{typed} -> {_truncate(a.get('result','?'), 60)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PLANNER PROMPT
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are the PLANNER component of a two-LLM autonomous browser agent.

YOUR ROLE:
  - Study the screenshot, OCR text, and DOM summary provided.
  - Understand the current UI state (page, modal, form, etc.).
  - Infer the overall workflow and identify the next logical step.
  - Return a structured JSON plan that describes exactly ONE next action.

YOU ARE NOT the executor. Do not worry about HOW the click will be performed
at the pixel level — focus on WHAT should happen and WHY.

The goal given to you may be brief (e.g. "Create a Service Note").
YOU are responsible for figuring out the intermediate steps by reading the UI.

AVAILABLE ACTIONS (for next_action.action):
  "click"           — click a button, link, checkbox, or UI element
  "type"            — type text into the currently focused field
  "select"          — choose from a dropdown / combobox / MUI Select
  "hover"           — hover to expand a menu
  "find_in_sidebar" — compound: expand sidebar, scroll it, click a nav item
  "scroll"          — scroll page or sidebar
  "wait"            — wait 2 s for UI to settle
  "done"            — task is confirmed complete (visible success message)

RESPOND WITH ONLY THE FOLLOWING JSON — no markdown, no extra keys:

{
  "status":        "acting",
  "page_context":  "<one of: login|post_login_nav|service_note_list|create_modal|form_filling|unknown>",
  "workflow_step": "<short label: what phase of the task we are in>",
  "next_action": {
    "action":             "<action name>",
    "target_description": "<exact visible text label>",
    "input_text":         "",
    "scroll_direction":   "down",
    "scroll_target":      "page"
  },
  "reasoning": "<1-3 sentences: why this is the correct next step>"
}

RULES:
1. Set status "done" ONLY when a clear confirmation (toast/banner/redirect) is visible.
2. Set status "failed" only if the task is impossible given what you see.
3. For dropdowns use action "select"; target_description = label text; input_text = option or "".
4. If the left navigation sidebar is collapsed or hover-activated, ALWAYS hover over it before searching.
5. Never scroll a sidebar until it has been expanded.
6. "find_in_sidebar" means:
   - Hover over the sidebar.
   - Wait for it to expand.
   - Scroll INSIDE the sidebar only.
   - Search after every scroll.
   - Click only when the requested item is visible.
7. Never repeat an action that already succeeded (check history carefully).
8. Work top-to-bottom on forms; fill required (*) fields before submitting.
9. Modal open? Do NOT scroll or navigate — interact only with the modal.
10. Login page: click username field → type username → click password field → type password → click submit.
11. Do NOT re-type already-filled fields (see history).

CRITICAL rules:
- ALWAYS click a field before typing into it. Never type without clicking first.
- Check history: if the last action was click on Username, next must be type into Username.
- Never type into a field that is not currently focused.
- Do not repeat a step that already succeeded (check history).
- Do not click Sign In until both username and password have been typed.
"""


def _build_planner_prompt(
    goal:           str,
    history:        list,
    ocr_text:       str,
    dom_summary:    str,
    screen_size:    tuple,
) -> str:
    history_text = _format_history(history)
    w, h         = screen_size

    return f"""GOAL (brief, your job is to figure out the steps):
{goal.strip()}

SCREEN: {w} x {h} px

RECENT ACTION HISTORY (last 8):
{history_text}

OCR TEXT VISIBLE ON SCREEN (words detected by OCR, left-to-right order):
{ocr_text if ocr_text else "(no OCR text available)"}

DOM SUMMARY (key interactive elements found in the live page):
{dom_summary if dom_summary else "(no DOM summary available)"}

---
Look at the screenshot carefully.
Identify the current page/modal state, infer the workflow, and choose the single best next action.
"""


# ---------------------------------------------------------------------------
# DOM SUMMARY EXTRACTOR  (called by agent.py via BrowserController)
# ---------------------------------------------------------------------------

def extract_dom_summary(driver) -> str:
    """
    Extract a compact summary of the live DOM — interactive elements only.
    This is injected into the Planner prompt so it can reason about element
    labels without depending solely on OCR or pixel vision.

    Runs entirely in the browser via JavaScript; no network call needed.
    Returns a plain-text summary (max ~800 chars).
    """
    try:
        summary = driver.execute_script("""
            var MAX_ITEMS = 40;
            var items     = [];

            // Priority selectors: interactive + labelled elements
            var selectors = [
                'button:not([disabled])',
                'input:not([type="hidden"])',
                'select',
                'textarea',
                '[role="button"]',
                '[role="combobox"]',
                '[role="listbox"]',
                '[role="option"]',
                '[role="menuitem"]',
                'a[href]',
                'label',
                '[aria-label]',
                '[placeholder]',
            ];

            var seen = new Set();
            for (var s = 0; s < selectors.length && items.length < MAX_ITEMS; s++) {
                var els = document.querySelectorAll(selectors[s]);
                for (var i = 0; i < els.length && items.length < MAX_ITEMS; i++) {
                    var el   = els[i];
                    var rect = el.getBoundingClientRect();
                    // skip hidden / off-screen elements
                    if (rect.width < 2 || rect.height < 2) continue;
                    if (rect.bottom < 0 || rect.top > window.innerHeight) continue;

                    var key = el.tagName + '|' + (el.innerText||'').trim().substring(0,30);
                    if (seen.has(key)) continue;
                    seen.add(key);

                    var text  = (el.innerText       || '').trim().substring(0, 50);
                    var label = (el.getAttribute('aria-label') ||
                                 el.getAttribute('placeholder') ||
                                 el.getAttribute('title')      || '').trim().substring(0, 50);
                    var tag   = el.tagName.toLowerCase();
                    var type  = el.getAttribute('type') || '';
                    var role  = el.getAttribute('role') || '';

                    var desc = tag;
                    if (type)  desc += '[' + type + ']';
                    if (role)  desc += '[role=' + role + ']';
                    if (label) desc += ' label="' + label + '"';
                    if (text && text !== label) desc += ' text="' + text + '"';

                    items.push(desc);
                }
            }
            return items.join(' | ');
        """)
        if summary:
            return summary[:1200]   # hard cap so we don't blow the prompt
        return "(empty DOM summary)"
    except Exception as e:
        logger.debug(f"[DOM] extract_dom_summary failed: {e}")
        return "(DOM summary unavailable)"


# ---------------------------------------------------------------------------
# GEMINI PLANNER
# ---------------------------------------------------------------------------

class GeminiPlanner:
    """
    Planner LLM backed by Google Gemini.

    Accepts any vision-capable Gemini model (e.g. gemini-2.5-flash).
    Temperature is intentionally higher than the Executor (default 0.6)
    to allow creative recovery reasoning when the UI is unexpected.

    Interface:
        plan = planner.plan(
            screenshot     = PIL.Image,
            goal           = str,
            action_history = list[dict],
            ocr_text       = str,           # optional; "" if not available
            dom_summary    = str,           # optional; "" if not available
        )
        # Returns: ExecutionPlan dict (see module docstring)
    """

    def __init__(
        self,
        api_key:     str,
        model_name:  str   = "gemini-2.5-flash",
        temperature: float = 0.6,
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
            raise RuntimeError(f"GeminiPlanner init failed: {e}")
        logger.info(f"GeminiPlanner ready: {self.model_name}  temp={self.temperature}")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def plan(
        self,
        screenshot:     Image.Image,
        goal:           str,
        action_history: list,
        ocr_text:       str = "",
        dom_summary:    str = "",
    ) -> dict:
        """
        Analyse the current browser state and return an ExecutionPlan.

        Parameters
        ----------
        screenshot     : PIL screenshot of the current browser window
        goal           : High-level goal string (brief is fine)
        action_history : Raw list of action dicts from agent memory
        ocr_text       : Flat string of OCR words detected on screen
        dom_summary    : Compact string of interactive DOM elements

        Returns
        -------
        dict  ExecutionPlan (see module docstring for schema)
        """
        img_bytes = self._pil_to_jpeg_bytes(screenshot)
        prompt    = _build_planner_prompt(
            goal        = goal,
            history     = action_history,
            ocr_text    = ocr_text,
            dom_summary = dom_summary,
            screen_size = screenshot.size,
        )
        raw  = self._call_gemini(img_bytes, prompt)
        plan = _parse_plan(raw)
        logger.info(json.dumps(plan, indent=2))
        logger.info(
            f"[PLANNER] page={plan.get('page_context','?')} "
            f"step='{plan.get('workflow_step','?')[:60]}' "
            f"action=[{plan.get('next_action',{}).get('action','?').upper()}] "
            f"target='{plan.get('next_action',{}).get('target_description','?')[:50]}'"
        )
        return plan

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
                    system_instruction = _PLANNER_SYSTEM,
                    temperature        = self.temperature,
                    max_output_tokens  = 1024,
                    response_mime_type = "application/json",
                ),
            )
            raw = response.text or ""
            logger.debug(f"[PLANNER] raw (first 400):\n{raw[:400]}")
            return raw

        except Exception as e:
            logger.error(f"[PLANNER] Gemini call failed: {e}")
            # MIME type fallback for older model versions
            if "mime" in str(e).lower() or "application/json" in str(e).lower():
                return self._call_gemini_text_plain(img_bytes, user_prompt)
            return ""

    def _call_gemini_text_plain(self, img_bytes: bytes, user_prompt: str) -> str:
        """Fallback: text/plain MIME (some older model variants)."""
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model    = self.model_name,
                contents = [
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    user_prompt,
                ],
                config = types.GenerateContentConfig(
                    system_instruction = _PLANNER_SYSTEM,
                    temperature        = self.temperature,
                    max_output_tokens  = 1024,
                    response_mime_type = "text/plain",
                ),
            )
            return response.text or ""
        except Exception as e:
            logger.error(f"[PLANNER] text/plain fallback also failed: {e}")
            return ""


# ---------------------------------------------------------------------------
# PLAN PARSER
# ---------------------------------------------------------------------------

def _parse_plan(raw: str) -> dict:
    """
    Extract and validate the ExecutionPlan JSON from the Planner's raw output.
    Applies the same robust repair logic used in vision_model._parse_response.
    """
    if not raw or not raw.strip():
        logger.warning("[PLANNER] Empty response — using scroll fallback plan.")
        return _fallback_plan("empty planner response")

    cleaned = raw
    cleaned = re.sub(r"<think>.*?</think>",         "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?",               "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("`").strip()

    start = cleaned.find("{")
    if start == -1:
        logger.warning(f"[PLANNER] No JSON in response: {raw[:200]}")
        return _fallback_plan("no JSON found")

    json_candidate = cleaned[start:]

    # Try full parse first
    last_brace = json_candidate.rfind("}")
    if last_brace != -1:
        try:
            parsed = json.loads(json_candidate[:last_brace + 1])
            return _fill_plan_defaults(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Truncation repair: walk back from end
    pos = len(json_candidate) - 1
    while pos >= 0:
        pos = json_candidate.rfind("}", 0, pos)
        if pos == -1:
            break
        try:
            parsed = json.loads(json_candidate[:pos + 1])
            logger.info(f"[PLANNER] Truncation repair succeeded at pos {pos}")
            return _fill_plan_defaults(parsed)
        except (json.JSONDecodeError, ValueError):
            pos -= 1

    logger.warning(f"[PLANNER] All parse attempts failed. Raw: {raw[:300]}")
    return _fallback_plan("parse error")


def _fill_plan_defaults(parsed: dict) -> dict:
    """Ensure all required keys exist with safe defaults."""
    parsed.setdefault("status",        "acting")
    parsed.setdefault("page_context",  "unknown")
    parsed.setdefault("workflow_step", "unknown step")
    parsed.setdefault("reasoning",     "")

    # Ensure next_action sub-dict is present and complete
    na = parsed.setdefault("next_action", {})
    na.setdefault("action",             "scroll")
    na.setdefault("target_description", "page")
    na.setdefault("input_text",         "")
    na.setdefault("scroll_direction",   "down")
    na.setdefault("scroll_target",      "page")

    return parsed


def _fallback_plan(reason: str) -> dict:
    """Safe default plan: scroll down to gather more context."""
    return {
        "status":       "acting",
        "page_context": "unknown",
        "workflow_step": f"recovery scroll ({reason})",
        "next_action": {
            "action":             "scroll",
            "target_description": "page",
            "input_text":         "",
            "scroll_direction":   "down",
            "scroll_target":      "page",
        },
        "reasoning": f"Planner could not parse its output ({reason}). Scrolling to re-observe.",
    }