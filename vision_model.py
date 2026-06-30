import json
import re
import logging
import io
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

# 
# VISION MODEL  Gemini Flash 2.5 edition
#
# Uses Google Gemini Flash 2.5 as the reasoning brain.
# Gemini can see the ACTUAL screenshot pixels (not just OCR text),
# making it much more accurate than the OCR-only approach.
#
# Two modes work together:
#   1. Gemini Flash 2.5   looks at screenshot  decides action
#   2. pytesseract OCR    finds the (x,y) coordinates to click
#      (OCR is still used by executor.py to resolve WHERE to click)
#
# Setup:
#   pip install google-genai pytesseract
#   Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
#   API key: https://aistudio.google.com/apikey
# 

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# 
# FACTORY  called from main.py
# 

def create_vision_model(
    mode: str = "gemini",
    gemini_api_key: Optional[str] = None,
    gemini_model:   str = "gemini-2.5-flash",
    # kept for backwards compat, ignored when mode=gemini
    gemma_model_name: str = "",
    hf_api_token:     Optional[str] = None,
    openai_api_key:   Optional[str] = None,
):
    """
    Factory function. Call this from main.py to get a vision model.

    Args:
        mode:           "gemini" (recommended) | "ocr" | "gemma" | "hf_api"
        gemini_api_key: Your Google AI Studio API key
        gemini_model:   Model name  "gemini-2.5-flash" or "gemini-2.0-flash"
    """
    mode = mode.lower().strip()

    if mode == "gemini":
        if not gemini_api_key:
            raise ValueError(
                "Gemini mode requires an API key.\n"
                "Get one free at: https://aistudio.google.com/apikey\n"
                "Then set GEMINI_API_KEY in main.py"
            )
        return GeminiVisionModel(
            api_key=gemini_api_key,
            model_name=gemini_model,
        )

    elif mode == "ocr":
        return OCRFallbackModel(
            hf_api_token=hf_api_token,
            openai_api_key=openai_api_key,
        )

    elif mode == "gemma":
        return LocalVisionModel(model_name=gemma_model_name)

    elif mode == "hf_api":
        if not hf_api_token:
            raise ValueError("hf_api mode requires a HuggingFace API token.")
        return HFAPIVisionModel(api_token=hf_api_token)

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Choose: 'gemini', 'ocr', 'gemma', 'hf_api'")


# 
# SHARED: PROMPT BUILDER
# 


def _derive_form_state(history: list) -> dict:
    """
    FIX 1+5: Scan raw history dicts to build ground-truth form state.

    Returns a dict describing what has actually been typed into each field,
    derived from the sequence of (click target -> type input_text) pairs.
    This gives the LLM verified facts instead of letting it guess.

    Example return:
        {
            "username_filled": True,
            "username_value":  "aSato",
            "password_filled": False,
            "password_value":  "",
            "login_clicked":   False,
        }
    """
    state = {
        "username_filled": False,
        "username_value":  "",
        "password_filled": False,
        "password_value":  "",
        "login_clicked":   False,
    }

    username_keywords = {"username", "user", "email", "login", "id", "account"}
    password_keywords = {"password", "pass", "pwd", "secret"}
    submit_keywords   = {"login", "sign in", "log in", "submit", "signin"}

    last_clicked_field = None   # tracks which field was most recently focused

    for entry in history:
        action = entry.get("action", "").lower()
        target = entry.get("target_description", "").lower()
        typed  = entry.get("input_text", "")
        result = entry.get("result", "")

        if action == "click" and "success" in result:
            # Determine which field type was clicked
            if any(k in target for k in username_keywords):
                last_clicked_field = "username"
            elif any(k in target for k in password_keywords):
                last_clicked_field = "password"
            elif any(k in target for k in submit_keywords):
                state["login_clicked"] = True
                last_clicked_field = None

        elif action == "type" and typed and "success" in result:
            # Attribute the typed text to the last focused field
            if last_clicked_field == "username":
                state["username_filled"] = True
                state["username_value"]  = typed
            elif last_clicked_field == "password":
                state["password_filled"] = True
                state["password_value"]  = typed

    return state


def _derive_login_substep(form_state: dict) -> str:
    """
    Convert form state into an explicit, deterministic instruction
    for the LLM about exactly which login step to perform next.
    This prevents the LLM from skipping steps or re-doing completed ones.

    NOTE: Once login is submitted, return an empty string so the post-login
    goal instructions are not confused by stale "CURRENT LOGIN STEP" text.
    """
    if form_state["login_clicked"]:
        # Login already submitted -- do NOT inject any login-step instruction.
        # The LLM should now focus purely on the post-login goal.
        return ""

    if not form_state["username_filled"]:
        return (
            "CURRENT LOGIN STEP: Click the username/email input field. "
            "The username has NOT been typed yet."
        )
    if not form_state["password_filled"]:
        return (
            f"CURRENT LOGIN STEP: Click the password input field and type the password. "
            f"Username '{form_state['username_value']}' is already filled. "
            f"Do NOT re-type the username."
        )
    return (
        "CURRENT LOGIN STEP: Click the Login/Sign In submit button. "
        "Both username and password fields are already filled. "
        "Do NOT click any field -- click the submit button only."
    )


def _detect_page_context(history: list) -> str:
    """
    Analyse the last few history entries to determine what UI state
    the agent is currently in. This drives context-specific prompt sections.

    Returns one of:
        "login"               - on the login page, not yet submitted
        "post_login_nav"      - logged in, need to navigate to Service Note
        "service_note_list"   - on the Service Note list page
        "create_modal"        - the Create Service Note modal is open
        "form_filling"        - on the full service note form after Proceed
        "done"                - task complete
    """
    if not history:
        return "login"

    raw = [a for a in history if isinstance(a, dict)]
    if not raw:
        return "login"

    # Check login state
    form_state = _derive_form_state(raw)
    if not form_state["login_clicked"]:
        return "login"

    # Walk history from newest to oldest looking for page signals
    for entry in reversed(raw):
        action = entry.get("action", "").lower()
        target = entry.get("target_description", "").lower()
        result = entry.get("result", "").lower()

        if "task_complete" in result or entry.get("status") == "done":
            return "done"

        # After clicking Proceed in the modal we enter the full form
        if action == "click" and "proceed" in target and "success" in result:
            return "form_filling"

        # Modal is open when we just clicked "Create Service Note"
        if action == "click" and "create service note" in target and "success" in result:
            return "create_modal"

        # On the Service Note list if find_in_sidebar for Service Note succeeded
        if action == "find_in_sidebar" and "service note" in target and "success" in result:
            return "service_note_list"

        # Still navigating to Service Note
        if action == "find_in_sidebar" and "service note" in target:
            return "post_login_nav"

    return "post_login_nav"


def _build_context_block(page_context: str) -> str:
    """
    Return context-specific guidance injected into the prompt
    based on which UI state the agent is currently in.
    """

    if page_context == "login":
        return ""  # handled separately by login_block

    if page_context == "service_note_list":
        return """
PAGE CONTEXT — SERVICE NOTE LIST
You are on the Service Note list page. You can see:
  - A table of existing service notes (DIS-NTF-XXXX rows)
  - A "+ Create Service Note" button in the TOP-RIGHT corner of the page (dark navy button)
  - A search bar on the left

YOUR IMMEDIATE NEXT ACTION:
  → Click the "+ Create Service Note" button in the top-right corner.
  Use action "click" with target_description "+ Create Service Note".
  This button is clearly visible; do NOT scroll or navigate first.
"""

    if page_context == "create_modal":
        return """
PAGE CONTEXT — CREATE SERVICE NOTE MODAL
A modal dialog has opened with these elements:
  - Toggle: "Approved RA Bill" (leave OFF / disabled unless you see it is required)
  - Dropdown: "Business Unit *" (required — marked with *)
  - Dropdown: "PO No. *"        (required — marked with *)
  - Dropdown: "Service Header *" (required — marked with *)
  - Radio option: "Service Note Without Dynamic Headers" (blue dot — default)
  - Radio option: "Service Note With Dynamic Headers"   (orange dot)
  - Buttons: "Cancel" and "Proceed"

YOUR TASK — fill all required dropdowns then click Proceed:

STEP A: Select "Business Unit" dropdown first.
  action = "select"
  target_description = "Business Unit"
  input_text = ""   ← leave blank; executor will pick the first available option

STEP B: Select "PO No." dropdown.
  action = "select"
  target_description = "PO No."
  input_text = ""   ← leave blank; executor will pick the first available option

STEP C: Select "Service Header" dropdown.
  action = "select"
  target_description = "Service Header"
  input_text = ""   ← leave blank; executor will pick the first available option

STEP D: Click "Proceed".
  action = "click"
  target_description = "Proceed"

RULES FOR THIS MODAL:
- Fill dropdowns ONE AT A TIME, in order A → B → C → D.
- Check the recent action history to know which step you are on — do NOT repeat a completed step.
- Do NOT click "Cancel".
- Do NOT try to scroll or navigate while the modal is open.
- If a dropdown shows no options, select it anyway (executor handles it).
- Use action "select" (NOT "click") for all three dropdowns.
"""

    if page_context == "form_filling":
        return """
PAGE CONTEXT — SERVICE NOTE FORM
After clicking Proceed, a full service note creation form has loaded.
You are now in FORM-FILLING MODE.

HOW TO APPROACH THIS FORM:
1. SCAN: Look at the screenshot carefully. Identify ALL visible fields and their labels.
2. PRIORITISE: Fill required fields (marked with * or highlighted) before optional ones.
3. ORDER: Work top-to-bottom, left-to-right through the form.
4. VALUES: Generate realistic but generic values:
   - Text fields: use short descriptive text, e.g. "Standard service delivery"
   - Date fields: use today's date or a near-future date
   - Number fields: use "1" as a safe default
   - Dropdowns: pick the first available option if you are unsure
5. SUBMIT: Once ALL required fields are filled, look for a "Save", "Submit",
   "Create", or "Confirm" button and click it.
6. CONFIRMATION: After submitting, wait one step to confirm success
   (look for a success banner, toast message, or redirect to the list page).
   Then set status = "done".

FIELD ACTION RULES:
  - Text/number inputs   → action "click" the field first, then action "type" the value
  - Dropdowns/selects    → action "select" with target_description = label, input_text = option
  - Date pickers         → action "click" the field, then action "type" the date string
  - Checkboxes/toggles   → action "click" the element
  - Searchable dropdowns → action "click" to open, then action "type" to filter, then action "click" the result

DO NOT:
  - Hardcode any coordinates or element IDs
  - Click "Cancel" or navigate away from the form
  - Skip required fields
  - Submit before all required (*) fields are filled
"""

    # post_login_nav or unknown
    return """
PAGE CONTEXT — POST-LOGIN NAVIGATION
You are logged in. Your next goal is to reach the Service Note section.
Use action "find_in_sidebar" with target_description "Service Note".
This single compound action will expand the sidebar, scroll it, and click the item.
"""


def _build_prompt(goal: str, history: list, screen_size: tuple = None) -> str:
    """
    Builds the full LLM prompt with:
    - Goal
    - Recent action history
    - Login state grounding (during login only)
    - Page-context-specific guidance (post-login)
    - Action schema and rules
    """
    # --- History display ---
    if history:
        recent = history[-5:]
        history_lines = []
        for i, a in enumerate(recent):
            if isinstance(a, dict):
                typed_note = f" | typed: '{a['input_text']}'" if a.get("input_text") else ""
                history_lines.append(
                    f"  {i+1}. [{a.get('action','?').upper()}] "
                    f"'{a.get('target_description','?')}'"
                    f"{typed_note} -> {a.get('result','?')}"
                )
            else:
                history_lines.append(f"  {i+1}. {a}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "  (none - this is the first step)"

    # --- Form state grounding (login phase) ---
    raw_dicts     = [a for a in history if isinstance(a, dict)]
    form_state    = _derive_form_state(raw_dicts)
    login_substep = _derive_login_substep(form_state)

    if not form_state["login_clicked"]:
        form_state_text = (
            f"  username_filled : {form_state['username_filled']}"
            + (f" (value: '{form_state['username_value']}')" if form_state['username_filled'] else "")
            + f"\n  password_filled : {form_state['password_filled']}"
            + f"\n  submit_clicked  : {form_state['login_clicked']}"
        )
        login_block = f"""
VERIFIED FORM STATE (ground truth - do NOT contradict this):
{form_state_text}

{login_substep}
"""
    else:
        login_block = "LOGIN STATUS: Completed. Focus on the post-login goal above.\n"

    # --- Page context block ---
    page_context   = _detect_page_context(history)
    context_block  = _build_context_block(page_context)

    size_hint = ""
    if screen_size:
        w, h      = screen_size
        size_hint = f"\nScreen resolution: {w} x {h} pixels."

    return f"""You are an autonomous browser agent controlling a real web browser.
You are shown a screenshot of the current browser state.
Your job is to decide ONE action to take next to complete the goal.{size_hint}

GOAL:
{goal.strip()}

RECENT ACTIONS (last 5):
{history_text}

{login_block}
{context_block}

=== LOOK AT THE SCREENSHOT CAREFULLY ===
Identify the current page / modal, then choose the SINGLE BEST next action.

RESPOND WITH ONLY THIS JSON — no explanation outside the braces:

{{
  "status": "acting",
  "action": "click",
  "target_description": "the exact visible text label to interact with",
  "input_text": "",
  "scroll_direction": "down",
  "scroll_target": "page",
  "reasoning": "brief explanation of why"
}}

ACTION FIELD VALUES:

"status": "acting" | "done" | "failed"

"action":
  "click"           — click a button, link, or UI element
  "type"            — type text into the currently focused field
  "select"          — choose an option from a dropdown / combobox / MUI Select
  "hover"           — hover the mouse over an element (e.g. to expand a menu)
  "find_in_sidebar" — compound: expand sidebar, scroll it, and click a nav item
  "scroll"          — scroll the page or sidebar
  "wait"            — wait 2 seconds for the UI to settle
  "navigate"        — go to a specific URL

"target_description":
  The EXACT visible text label of the element you want to interact with.
  Examples: "+ Create Service Note", "Business Unit", "Proceed", "Save"
  For scroll/wait: write "page"
  For find_in_sidebar: write the sidebar menu item name, e.g. "Service Note"

"input_text":
  For "type":     the text to type into the field
  For "select":   the option to choose (leave "" to auto-pick the first option)
  For "navigate": the full URL (must start with http:// or https://)
  Leave "" for all other actions

"scroll_direction": "down" | "up"  (only used when action = "scroll")

"scroll_target":
  "page"    — scroll the main viewport (default)
  "sidebar" — scroll only the sidebar nav container

"reasoning": 1–2 sentence explanation of why you chose this action

=== RULES BY SITUATION ===

MODAL DIALOG IS OPEN:
1. Never click Cancel unless the task says to cancel.
2. Fill required dropdowns (marked *) using action "select" one at a time.
3. After all required fields are filled, click "Proceed" or "Submit".
4. Do not scroll or navigate while a modal is open.

DROPDOWN / COMBOBOX:
- Always use action "select" (not "click") to set a dropdown value.
- target_description = the label text above/inside the dropdown.
- input_text = the option to pick, or "" to auto-pick the first option.
- If the dropdown opens a list, the executor handles selection automatically.

MAIN PAGE BUTTONS (not sidebar):
- "+ Create Service Note", "Proceed", "Save", "Submit", "Approve", "Reject"
- Use action "click" for these; they are never in the sidebar.

SIDEBAR NAVIGATION:
- Use action "find_in_sidebar" to reach any sidebar menu item.
- This single action handles hover, expand, scroll, and click internally.
- Do NOT use scroll + sidebar as the first attempt; try find_in_sidebar first.

LOGIN PAGE:
1. Follow CURRENT LOGIN STEP exactly — do not skip or repeat steps.
2. username_filled=True means do NOT touch the username field again.
3. password_filled=True means do NOT touch the password field again.
4. Click only "Login" / "Sign In" to submit — never "Forgot Password" or "Cancel".

FORM FILLING:
- Work top-to-bottom, fill all required (*) fields before submitting.
- Use "click" then "type" for text fields.
- Use "select" for dropdowns and comboboxes.
- Submit only after all required fields are filled.
- After submit, wait one step and check for a success message before setting status "done".

NAVIGATE ACTION:
- Only use for known URLs. Never set target_description to "/" or leave input_text blank.

GENERAL:
- Set status "done" only when a visible confirmation (toast, banner, redirect) confirms success.
- If you are unsure what page you are on, prefer "wait" to let the UI settle.
"""


# 
# SHARED: JSON PARSER
# 

def _parse_response(raw: str) -> dict:
    """
    Safely extract valid JSON from the LLM response.

    Handles:
      - <think>...</think> blocks (Gemini 2.5 Flash reasoning tokens)
      - Markdown fences: ```json ... ```
      - Extra text before/after the JSON object
      - TRUNCATED JSON: response cut off mid-object by max_output_tokens
        -> we find the last valid closing brace and try to complete the object
      - Missing optional keys (filled with safe defaults)
      - Completely unparseable output (returns scroll fallback)
    """
    if not raw or not raw.strip():
        logger.warning("Model returned empty response")
        return _fallback("empty response")

    cleaned = raw

    # --- Strip <think>...</think> reasoning blocks ---
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reflection>.*?</reflection>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()

    # --- Remove markdown code fences ---
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("`").strip()

    logger.debug(f"Cleaned response (first 400 chars):\n{cleaned[:400]}")

    # --- Find the start of the JSON object ---
    start = cleaned.find("{")
    if start == -1:
        logger.warning(f"No JSON object found in response. Raw (first 300): {raw[:300]}")
        return _fallback("no JSON object found")

    json_candidate = cleaned[start:]

    # --- Try to parse the full string first (happy path) ---
    # Find the last } to handle any trailing text after the JSON
    last_brace = json_candidate.rfind("}")
    if last_brace != -1:
        try:
            parsed = json.loads(json_candidate[:last_brace + 1])
            return _fill_defaults_and_log(parsed)
        except (json.JSONDecodeError, ValueError):
            pass  # fall through to truncation repair

    # --- TRUNCATION REPAIR ---
    # The response was cut off mid-object (max_output_tokens hit).
    # Strategy: walk backwards from the end of the string, try progressively
    # shorter substrings ending at each '}', stop at the first valid parse.
    # This recovers partial JSON like:
    #   {"status":"acting","action":"scroll","target_description":"page",
    #    "input_text":"","scroll_direction":"down","scroll_target":"sidebar",
    #    "reasoning":"The sidebar is expanded but Servi    <-- CUT HERE
    # by finding the last complete key-value pair.
    logger.warning(
        f"Initial JSON parse failed -- attempting truncation repair. "
        f"Response length: {len(raw)} chars"
    )

    # Try each } position from right to left
    pos = len(json_candidate) - 1
    while pos >= 0:
        pos = json_candidate.rfind("}", 0, pos)
        if pos == -1:
            break
        try:
            parsed = json.loads(json_candidate[:pos + 1])
            logger.info(f"Truncation repair succeeded at pos {pos}/{len(json_candidate)}")
            return _fill_defaults_and_log(parsed)
        except (json.JSONDecodeError, ValueError):
            pos -= 1  # try the next } leftward
            continue

    # --- Last resort: inject a closing brace and try ---
    # Handles the case where the ONLY } is missing entirely (very short truncation).
    for suffix in ['"}', '"', "}}"]:
        try:
            attempt = json_candidate.rstrip() + suffix
            parsed = json.loads(attempt)
            logger.info(f"Truncation repair (suffix injection) succeeded")
            return _fill_defaults_and_log(parsed)
        except (json.JSONDecodeError, ValueError):
            continue

    logger.warning(f"All parse attempts failed. Raw (first 400):\n{raw[:400]}")
    return _fallback("parse error: truncated/malformed JSON")


def _fill_defaults_and_log(parsed: dict) -> dict:
    """Fill missing keys with safe defaults and log the chosen action."""
    parsed.setdefault("status",             "acting")
    parsed.setdefault("action",             "scroll")
    parsed.setdefault("target_description", "page")
    parsed.setdefault("input_text",         "")
    parsed.setdefault("scroll_direction",   "down")
    parsed.setdefault("scroll_target",      "page")
    parsed.setdefault("reasoning",          "")

    logger.info(
        f"Model -> [{parsed['action'].upper()}] "
        f"'{parsed['target_description'][:60]}'"
    )
    if parsed.get("reasoning"):
        logger.info(f"         {parsed['reasoning'][:100]}")
    return parsed




# 
# BACKEND 1: GEMINI FLASH 2.5   PRIMARY BACKEND
# 

class GeminiVisionModel:
    """
    Uses Google Gemini Flash 2.5 to look at screenshots and decide actions.

    Why Gemini Flash 2.5?
      - Native vision: sees the ACTUAL screenshot pixels
      - Fast: typically responds in 1-3 seconds
      - Cheap: very low cost per screenshot
      - Smart: understands complex UI layouts, forms, menus

    How it works:
      1. Convert PIL screenshot -> JPEG bytes
      2. Send to Gemini API with the task prompt
      3. Gemini looks at the image and returns JSON action
      4. executor.py uses pytesseract OCR to find click coordinates

    Requirements:
      pip install google-genai
      API key from https://aistudio.google.com/apikey
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key    = api_key
        self.model_name = model_name
        self._client    = None
        self._last_ocr  = []   # OCR bounding boxes used by executor.py
        self._setup()

    def _setup(self):
        """Initialize the Gemini client and verify the API key works."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Google GenAI SDK not installed.\n"
                "Run: pip install google-genai"
            )

        self._client = genai.Client(api_key=self.api_key)

        # Also set up pytesseract for coordinate resolution
        # (Gemini tells us WHAT to click; OCR tells us WHERE it is on screen)
        self._setup_tesseract()

        logger.info(f"GeminiVisionModel ready: {self.model_name}")

    def _setup_tesseract(self):
        """Configure pytesseract for coordinate lookup."""
        import os
        try:
            import pytesseract
            if os.path.isfile(TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            # If file not found, pytesseract tries system PATH automatically
        except ImportError:
            logger.warning(
                "pytesseract not installed -- coordinate lookup will use "
                "direction hints only.\n"
                "Install with: pip install pytesseract"
            )

    def analyze(
        self,
        screenshot:     Image.Image,
        goal:           str,
        action_history: list,
    ) -> dict:
        """
        Send screenshot to Gemini and get the next action.

        Args:
            screenshot:     PIL Image of current browser state
            goal:           Plain English task description
            action_history: List of recent action dicts

        Returns:
            dict: {status, action, target_description, input_text, reasoning}
        """
        # Step 1: Run OCR to get bounding boxes for executor
        self._last_ocr = self._run_ocr(screenshot)
        logger.info(f"OCR: {len(self._last_ocr)} text regions found")

        # Step 2: Convert screenshot to JPEG bytes for Gemini API
        img_bytes = self._pil_to_jpeg_bytes(screenshot)

        # Step 3: Build prompt
        prompt = _build_prompt(goal, action_history, screenshot.size)

        # Step 4: Call Gemini API
        raw_response = self._call_gemini(img_bytes, prompt)

        # Step 5: Parse JSON from response
        return _parse_response(raw_response)

    def _call_gemini(self, img_bytes: bytes, prompt: str) -> str:
        """
        Make the actual API call to Gemini.

        Gemini accepts a list of "parts" -- we send:
          Part 1: The screenshot image (JPEG bytes)
          Part 2: The instruction prompt (text)

        The model sees both and responds with text (our JSON action).

        FIX: Use response_mime_type="application/json" to force Gemini to
        return only JSON without any reasoning preamble or markdown fences.
        This is the cleanest way to prevent the thinking-token JSON parse errors.
        """
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,          # 0 = deterministic output (best for JSON)
                    # FIX: raised from 1024 to 2048 -- the JSON output was being
                    # truncated mid-object at 1024 tokens, producing an unclosed {
                    # that _parse_response correctly rejected as "no JSON object found".
                    max_output_tokens=2048,
                    # application/json forces the model to output ONLY valid JSON.
                    # This prevents thinking tokens / reasoning prefixes from polluting
                    # the response and causing JSON parse failures.
                    response_mime_type="application/json",
                ),
            )

            # FIX: When response_mime_type="application/json" is set, response.text
            # contains only the JSON string. But some model versions still wrap it,
            # so we pass through _parse_response which strips any remaining cruft.
            raw = response.text
            if raw:
                logger.debug(f"Gemini raw response (first 400):\n{raw[:400]}")
            else:
                logger.warning("Gemini returned empty response.text")
            return raw or ""

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            # Common errors and solutions:
            if "API_KEY" in str(e).upper() or "401" in str(e):
                logger.error(
                    "Invalid API key. Get yours at: https://aistudio.google.com/apikey"
                )
            elif "QUOTA" in str(e).upper() or "429" in str(e):
                logger.error(
                    "Rate limit hit. Wait a minute and retry, or check your quota."
                )
            elif "mime" in str(e).lower() or "application/json" in str(e).lower():
                # Some older model versions don't support application/json MIME type.
                # Fall back to plain text and rely on _parse_response stripping.
                logger.warning(
                    "application/json MIME type not supported by this model version. "
                    "Retrying with text/plain..."
                )
                return self._call_gemini_text_plain(img_bytes, prompt)
            elif "model" in str(e).lower():
                logger.error(
                    f"Model '{self.model_name}' not found. "
                    "Try 'gemini-2.0-flash' or 'gemini-1.5-flash'."
                )
            return ""

    def _call_gemini_text_plain(self, img_bytes: bytes, prompt: str) -> str:
        """
        Fallback Gemini call using text/plain MIME type.
        Used when the model version doesn't support application/json.
        The response may contain thinking tokens -- _parse_response handles stripping.
        """
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=2048,
                    response_mime_type="text/plain",
                ),
            )
            return response.text or ""
        except Exception as e:
            logger.error(f"Gemini text/plain fallback also failed: {e}")
            return ""

    def _pil_to_jpeg_bytes(self, image: Image.Image) -> bytes:
        """
        Convert PIL Image to JPEG bytes for the Gemini API.

        We resize to max 1280px wide to:
          - Stay within Gemini's image size limits
          - Reduce token usage (cheaper + faster)
          - Still keep enough detail to read UI text
        """
        max_width = 1280
        if image.width > max_width:
            ratio  = max_width / image.width
            image  = image.resize(
                (max_width, int(image.height * ratio)),
                Image.LANCZOS
            )

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _run_ocr(self, screenshot: Image.Image) -> list:
        """
        Run pytesseract on the screenshot to get text bounding boxes.

        This gives executor.py the (x, y, w, h) of each word so it
        can click at the right coordinates when Gemini says "click Login".

        Returns:
            List of dicts: [{text, conf, x, y, w, h}, ...]
        """
        try:
            import pytesseract
        except ImportError:
            return []   # OCR not available -- executor will use XPath fallback

        try:
            tsv = pytesseract.image_to_data(
                screenshot,
                output_type=pytesseract.Output.DICT,
                config="--psm 11",   # sparse text mode -- good for UIs
            )

            results = []
            for i in range(len(tsv["text"])):
                word = tsv["text"][i].strip()
                conf = int(tsv["conf"][i])
                if not word or conf < 30:
                    continue
                results.append({
                    "text": word,
                    "conf": conf,
                    "x":    int(tsv["left"][i]),
                    "y":    int(tsv["top"][i]),
                    "w":    int(tsv["width"][i]),
                    "h":    int(tsv["height"][i]),
                })
            return results

        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return []

    def get_last_ocr_results(self) -> list:
        """
        Called by executor.py to get bounding boxes for coordinate clicks.
        Returns the OCR data from the most recent analyze() call.
        """
        return self._last_ocr


# 
# BACKEND 2: OCR FALLBACK (no Gemini API  uses pytesseract + text LLM)
# 

class OCRFallbackModel:
    """
    Lightweight fallback -- uses pytesseract for screen reading
    and a text LLM for reasoning.
    No vision model needed -- the LLM receives text, not pixels.
    """

    def __init__(
        self,
        hf_api_token:   Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.hf_api_token   = hf_api_token
        self.openai_api_key = openai_api_key
        self._last_ocr      = []
        self._setup_tesseract()
        self._verify_tesseract()

    def _setup_tesseract(self):
        import os
        try:
            import pytesseract
            if os.path.isfile(TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        except ImportError:
            pass

    def _verify_tesseract(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            logger.info("pytesseract ready (OCR fallback mode)")
        except ImportError:
            raise ImportError("pip install pytesseract")
        except Exception as e:
            raise RuntimeError(
                f"Tesseract binary not found: {e}\n"
                "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
            )

    def analyze(self, screenshot: Image.Image, goal: str, action_history: list) -> dict:
        import pytesseract, numpy as np

        tsv = pytesseract.image_to_data(
            screenshot, output_type=pytesseract.Output.DICT, config="--psm 11"
        )
        self._last_ocr = []
        text_items     = []
        for i in range(len(tsv["text"])):
            word = tsv["text"][i].strip()
            conf = int(tsv["conf"][i])
            if not word or conf < 30:
                continue
            self._last_ocr.append({
                "text": word, "conf": conf,
                "x": int(tsv["left"][i]),  "y": int(tsv["top"][i]),
                "w": int(tsv["width"][i]), "h": int(tsv["height"][i]),
            })
            text_items.append(f"'{word}'")

        screen_text = ", ".join(text_items) if text_items else "(no text detected)"
        w, h        = screenshot.size

        prompt = _build_prompt(goal, action_history, (w, h))
        prompt += f"\n\nSCREEN TEXT (from OCR): {screen_text}"

        raw = self._call_llm(prompt)
        return _parse_response(raw)

    def _call_llm(self, prompt: str) -> str:
        if self.openai_api_key:
            try:
                import openai
                client   = openai.OpenAI(api_key=self.openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512, temperature=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI failed: {e}")

        if self.hf_api_token:
            try:
                import requests
                resp = requests.post(
                    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
                    headers={"Authorization": f"Bearer {self.hf_api_token}"},
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 512, "return_full_text": False}},
                    timeout=90,
                )
                result = resp.json()
                return result[0].get("generated_text", "") if isinstance(result, list) else str(result)
            except Exception as e:
                logger.error(f"HF API failed: {e}")

        logger.warning("No LLM configured -- returning scroll fallback")
        return json.dumps(_fallback("no LLM configured"))

    def get_last_ocr_results(self) -> list:
        return self._last_ocr


# 
# BACKEND 3: LOCAL VISION MODEL (needs GPU)
# 

class LocalVisionModel:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        logger.info(f"Loading {model_name} locally (needs GPU)...")
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model     = LlavaForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, load_in_4bit=True,
            )
            logger.info(f"{model_name} loaded.")
        except ImportError:
            raise ImportError("pip install transformers torch accelerate bitsandbytes")

    def analyze(self, screenshot: Image.Image, goal: str, action_history: list) -> dict:
        import torch
        prompt = _build_prompt(goal, action_history, screenshot.size)
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text   = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=screenshot, text=text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        return _parse_response(self.processor.decode(new, skip_special_tokens=True))

    def get_last_ocr_results(self) -> list:
        return []


# 
# BACKEND 4: HUGGINGFACE INFERENCE API
# 

class HFAPIVisionModel:
    def __init__(self, api_token: str):
        self.api_token  = api_token
        self.model_name = "llava-hf/llava-1.5-7b-hf"

    def analyze(self, screenshot: Image.Image, goal: str, action_history: list) -> dict:
        import requests, base64
        buf     = io.BytesIO()
        screenshot.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        prompt  = _build_prompt(goal, action_history, screenshot.size)
        try:
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model_name}",
                headers={"Authorization": f"Bearer {self.api_token}"},
                json={"inputs": {"image": img_b64, "question": prompt},
                      "parameters": {"max_new_tokens": 512}},
                timeout=90,
            )
            resp.raise_for_status()
            r = resp.json()
            raw = r[0].get("generated_text", str(r)) if isinstance(r, list) else r.get("generated_text", str(r))
            return _parse_response(raw)
        except Exception as e:
            logger.error(f"HF API failed: {e}")
            return _fallback(f"API error: {e}")

    def get_last_ocr_results(self) -> list:
        return []