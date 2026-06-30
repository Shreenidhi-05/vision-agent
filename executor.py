import time
import logging
import re
from typing import Optional, Tuple

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Converts LLM action decisions into actual browser interactions.
    
    FIX 1: Login button detection now explicitly avoids "Forgot Password",
            "Sign Up", "Register" links  only targets real submit buttons.
    FIX 2: XPath matching is stricter  prefers <button> and <input type=submit>
            over generic <a> links when clicking login-related targets.
    """

    def __init__(self, browser_controller):
        self.browser = browser_controller
        self.driver  = browser_controller.driver
        self.actions = ActionChains(self.driver)

    # -------------------------------------------------------------------------
    # PUBLIC: execute()
    # -------------------------------------------------------------------------

    def execute(self, action_dict: dict, retry_count: int = 0) -> dict:
        action     = action_dict.get("action", "wait").lower().strip()
        target     = action_dict.get("target_description", "")
        input_text = action_dict.get("input_text", "")
        scroll_dir = action_dict.get("scroll_direction", "down")
        scroll_target = action_dict.get("scroll_target", "page")

        logger.info(f"Executing: [{action.upper()}]  Target: '{target}'")

        result = "unknown"
        try:
            if action == "click":
                result = self._execute_click(target)
            elif action == "hover":
                result = self._execute_hover(target)
            elif action == "type":
                result = self._execute_type(input_text, target)
            elif action == "scroll":
                result = self._execute_scroll(scroll_dir,scroll_target)
            elif action == "wait":
                result = self._execute_wait()
            elif action in ("done", "finish", "complete"):
                result = "task_complete"
            elif action == "navigate":
                url = input_text or target
                #  FIX: ignore navigate to "/" or empty  that's a Gemini
                #    hallucination when it can't figure out what to do next
                if not url or url.strip() in ("/", "", "page", "url"):
                    logger.warning(
                        f"Ignoring bad navigate target: '{url}'. "
                        "Scrolling instead to re-observe."
                    )
                    result = self._execute_scroll("down")
                else:
                    self.browser.navigate(url)
                    result = f"navigated to {url}"
            elif action == "find_in_sidebar":
                result = self._find_in_sidebar(target)
            elif action == "select":
                result = self._execute_select(target, input_text)
            else:
                logger.warning(f"Unknown action '{action}'  defaulting to wait")
                result = self._execute_wait()

        except Exception as e:
            logger.error(f"Action execution error: {e}")
            result = f"error: {str(e)[:120]}"

        action_dict["result"] = result
        logger.info(f"Result: {result}")
        return action_dict

    # -------------------------------------------------------------------------
    # CLICK  multi-strategy with login-aware filtering
    # -------------------------------------------------------------------------

    def _execute_click(self, target_description: str) -> str:
        screenshot = self.browser.take_screenshot()

        # Strategy 1: DOM/XPath lookup
        if self._click_by_xpath_text(target_description):
            logger.info("  [XPath] Click succeeded")
            time.sleep(1.5)
            return "success (XPath)"

        # Strategy 2: OCR lookup
        screenshot = self.browser.take_screenshot()
        coords = self._find_by_ocr_text(target_description, screenshot)
        if coords:
            x, y = coords
            logger.info(f"  [OCR] Clicking at ({x}, {y})")
            self._click_at_coordinates(x, y)
            time.sleep(1.5)
            return f"success (OCR at {x},{y})"

        # Strategy 3: Smart XPath (login-aware)
        if self._click_by_xpath_text(target_description):
            logger.info("  [XPath] Click succeeded")
            time.sleep(1.5)
            return "success (XPath fallback)"

        logger.warning(f"  All strategies failed for: '{target_description}'")
        return f"failed - not found: '{target_description}'"

    def _find_by_ocr_text(self, description: str, screenshot) -> Optional[Tuple[int, int]]:
        try:
            import pytesseract
            import numpy as np
        except ImportError:
            return None

        try:
            tsv = pytesseract.image_to_data(
                screenshot,
                output_type=pytesseract.Output.DICT,
                config="--psm 11",
            )
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return None

        # Words that don't identify a specific element
        stop_words = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to",
            "button", "link", "icon", "tab", "menu", "item", "field",
            "blue", "green", "red", "white", "black", "large", "small",
            "click", "press", "select", "find", "locate", "submit",
        }

        keywords = [
            w.lower() for w in re.split(r"\s+", description)
            if len(w) > 2 and w.lower() not in stop_words
        ]

        if not keywords:
            return None

        #  LOGIN-SPECIFIC FILTERING 
        # If target is a login/sign-in button, we must avoid matching
        # "Forgot Password", "Sign Up", "Register", "Cancel" etc.
        # We do this by scoring matches and penalising danger words.
        login_intent = any(k in keywords for k in ["login", "sign", "log"])
        danger_words = {"forgot", "password", "register", "signup",
                        "sign", "up", "cancel", "reset", "create", "new", "account"}

        best_match = None
        best_score = -999

        n = len(tsv["text"])
        for i in range(n):
            word = tsv["text"][i].strip()
            conf = int(tsv["conf"][i])
            if not word or conf < 30:
                continue

            wl    = word.lower()
            score = sum(1 for kw in keywords if kw in wl)

            # Exact match bonus
            if any(kw == wl for kw in keywords):
                score += 2

            # Penalty: if we're looking for login button, heavily penalise
            # any OCR word that looks like a "Forgot Password" link
            if login_intent and any(d in wl for d in danger_words):
                score -= 10

            if score > best_score:
                best_score = score
                x = int(tsv["left"][i])
                y = int(tsv["top"][i])
                w = int(tsv["width"][i])
                h = int(tsv["height"][i])
                cx = x + w // 2
                cy = y + h // 2
                best_match = (cx, cy)
                logger.debug(f"  OCR candidate: '{word}' score={score} at ({cx},{cy})")

        return best_match if best_score > 0 else None

    def _find_by_direction_hint(self, description: str) -> Optional[Tuple[int, int]]:
        desc  = description.lower()
        w     = self.browser.window_width
        h     = self.browser.window_height

        if any(word in desc for word in ["top", "upper", "header"]):
            y_ratio = 0.15
        elif any(word in desc for word in ["bottom", "lower", "footer"]):
            y_ratio = 0.85
        elif any(word in desc for word in ["middle", "center", "mid"]):
            y_ratio = 0.5
        else:
            y_ratio = None

        if any(word in desc for word in ["left", "sidebar", "nav"]):
            x_ratio = 0.15
        elif any(word in desc for word in ["right"]):
            x_ratio = 0.85
        elif any(word in desc for word in ["center", "middle"]):
            x_ratio = 0.5
        else:
            x_ratio = None

        if x_ratio is not None or y_ratio is not None:
            return (int((x_ratio or 0.5) * w), int((y_ratio or 0.5) * h))
        return None

    def _click_by_xpath_text(self, description: str) -> bool:
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        logger.info("Visible buttons:")

        for b in buttons:
            try:
                logger.info(f"BUTTON -> '{b.text}'")
            except:
                pass
        """
        XPath fallback  smarter ordering:
          1. Try <button> and <input type=submit> FIRST (more likely to be
             actual action buttons, not navigation links)
          2. Only fall back to <a> links if nothing else works
          3. Actively EXCLUDE elements containing danger words when
             clicking login-related targets
        """
        words = [
            w for w in re.split(r"\s+", description)
            if len(w) > 2 and w.isalpha()
        ]

        login_intent  = any(w.lower() in ("login", "signin", "log") for w in words)
        # Text fragments that indicate we've matched the wrong element
        exclude_texts = ["forgot", "password", "register", "sign up", "cancel", "reset"]

        for word in words[:3]:
            wl = word.lower()

            #  Ordered from most-specific to least-specific 
            # Buttons first (real action elements), links last
            xpaths = [
                # <button> elements
                f"//button[contains(translate(normalize-space(.), "
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), '{wl}')]",

                # <input type="submit"> or <input type="button">
                f"//input[@type='submit' and contains(translate(@value,"
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), '{wl}')]",

                # <input type="button">
                f"//input[@type='button' and contains(translate(@value,"
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), '{wl}')]",

                # Any element with matching text (links included  last resort)
                f"//*[contains(normalize-space(.),"
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), '{wl}')]",
            ]

            for xpath in xpaths:
                try:
                    elements = self.driver.find_elements(By.XPATH, xpath)
                    for elem in elements:
                        if not elem.is_displayed():
                            continue

                        elem_text = (elem.text or elem.get_attribute("value") or "").lower()

                        #  SAFETY CHECK 
                        # Skip this element if it contains a danger word and
                        # we are trying to click a login-related button
                        if login_intent and any(d in elem_text for d in exclude_texts):
                            logger.debug(
                                f"  Skipping '{elem_text[:40]}' "
                                f"(matches danger word filter)"
                            )
                            continue

                        try:
                            elem.click()
                            logger.debug(f"  XPath matched: '{elem_text[:40]}'")
                            return True
                        except Exception:
                            self.driver.execute_script("arguments[0].click();", elem)
                            return True
                except Exception:
                    continue

        return False

    def _click_at_coordinates(self, x: int, y: int):
        scroll_y   = max(0, y - self.browser.window_height // 2)
        viewport_y = y - scroll_y
        try:
            self.driver.execute_script(f"window.scrollTo(0, {scroll_y});")
            time.sleep(0.3)
        except Exception:
            viewport_y = y

        try:
            body = self.driver.find_element(By.TAG_NAME, "body")
            ActionChains(self.driver).move_to_element_with_offset(
                body, x, viewport_y
            ).click().perform()
            return
        except Exception as e:
            logger.debug(f"ActionChains click failed: {e}")

        try:
            self.driver.execute_script(
                f"var e=document.elementFromPoint({x},{viewport_y});"
                f"if(e){{e.click();e.dispatchEvent(new MouseEvent('click',{{bubbles:true}}))}}"
            )
        except Exception as e:
            logger.warning(f"JS click also failed at ({x},{viewport_y}): {e}")

    # -------------------------------------------------------------------------
    # HOVER
    # -------------------------------------------------------------------------

    def _execute_hover(self, target_description: str) -> str:
        # Special case for the left sidebar
        if any(k in target_description.lower() for k in ["sidebar", "module", "navigation", "nav"]):
            selectors = [
                "aside",
                "nav",
                "[role='navigation']",
                ".MuiDrawer-root",
                ".MuiDrawer-paper",
                ".sidebar",
                "#sidebar",
            ]

            for selector in selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    ActionChains(self.driver).move_to_element(element).pause(1).perform()
                    return "success"
                except Exception:
                    pass

        screenshot = self.browser.take_screenshot()
        coords = self._find_by_ocr_text(target_description, screenshot)

        if coords:
            x, y = coords
            try:
                element = self.driver.execute_script(
                    """
                    return document.elementFromPoint(arguments[0], arguments[1]);
                    """,
                    x,
                    y,
                )

                if element:
                    ActionChains(self.driver).move_to_element(element).pause(1).perform()
                    return "success"
            except Exception as e:
                return f"hover failed: {e}"

        return "hover failed"

    # -------------------------------------------------------------------------
    # TYPE
    # -------------------------------------------------------------------------

    def _execute_type(self, text: str, target_description: str = "") -> str:
        if not text:
            return "skipped - no text"

        element = None
        if target_description:
            try:
                element = self.driver.execute_script(
                    """
                    var desc = arguments[0].toLowerCase();
                    var inputs = document.querySelectorAll('input, textarea');
                    for (var i = 0; i < inputs.length; i++) {
                        var el = inputs[i];
                        if (el.offsetParent === null) continue;
                        var placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
                        var name = (el.getAttribute('name') || '').toLowerCase();
                        var id = (el.getAttribute('id') || '').toLowerCase();
                        var ariaLabel = (el.getAttribute('aria-label') || '').toLowerCase();
                        var labelText = '';
                        if (el.id) {
                            var lbl = document.querySelector('label[for="' + el.id + '"]');
                            if (lbl) labelText = lbl.innerText.toLowerCase();
                        }
                        if (placeholder.includes(desc) || name.includes(desc) ||
                            id.includes(desc) || ariaLabel.includes(desc) ||
                            labelText.includes(desc)) {
                            return el;
                        }
                    }
                    return null;
                    """,
                    target_description,
                )
            except Exception as e:
                logger.debug(f"DOM input search failed: {e}")

        if element is None:
            try:
                element = self.driver.switch_to.active_element
            except Exception:
                pass

        if element is None:
            return "type failed: could not find input element"

        try:
            try:
                element.click()
                time.sleep(0.2)
            except Exception:
                self.driver.execute_script("arguments[0].focus();", element)
                time.sleep(0.2)

            element.clear()
            time.sleep(0.1)
            for char in text:
                element.send_keys(char)
                time.sleep(0.04)
            return "success"
        except Exception:
            try:
                self.driver.execute_script(
                    """
                    var el = arguments[0];
                    var val = arguments[1];
                    el.focus();
                    var nativeSetter = Object.getOwnPropertyDescriptor(
                        window.HTMLInputElement.prototype, 'value'
                    ).set;
                    nativeSetter.call(el, val);
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                    """,
                    element,
                    text,
                )
                return "success"
            except Exception as fallback_error:
                logger.debug(f"Type fallback failed: {fallback_error}")
                return f"type failed: {str(fallback_error)[:80]}"

    # -------------------------------------------------------------------------
    # SCROLL
    # -------------------------------------------------------------------------

    def _execute_scroll(self, direction: str = "down", target: str = "page") -> str:
        pixels = 400 if direction == "down" else -400

        target = (target or "").lower()

        # Sidebar scrolling
        if any(word in target for word in [
            "sidebar",
            "module",
            "navigation",
            "nav"
        ]):
            try:
                sidebar = self.driver.execute_script("""
                    const candidates = [
                        ...document.querySelectorAll(
                            '.MuiDrawer-paper, .MuiDrawer-root, .sidebar, #sidebar, aside, nav, [role="navigation"]'
                        )
                    ];

                    for (const el of candidates) {
                        if (el.scrollHeight > el.clientHeight) {
                            return el;
                        }
                    }

                    return null;
                """)

                if sidebar:
                    before = self.driver.execute_script(
                        "return arguments[0].scrollTop;",
                        sidebar
                    )

                    self.driver.execute_script(
                        "arguments[0].scrollTop += arguments[1];",
                        sidebar,
                        pixels
                    )

                    after = self.driver.execute_script(
                        "return arguments[0].scrollTop;",
                        sidebar
                    )

                    logger.info(f"Sidebar scroll: {before} -> {after}")

                    time.sleep(0.7)
                    return f"scrolled sidebar {direction}"

            except Exception as e:
                logger.warning(f"Sidebar scroll failed: {e}")

        # Default page scroll
        self.driver.execute_script(
            "window.scrollBy(0, arguments[0]);",
            pixels
        )

        time.sleep(0.7)
        return f"scrolled page {direction}"
    # -------------------------------------------------------------------------
    # WAIT
    # -------------------------------------------------------------------------

    def _execute_wait(self, seconds: float = 2.0) -> str:
        time.sleep(seconds)
        try:
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except Exception:
            pass
        return f"waited {seconds}s"

    def _find_in_sidebar(self, target):
        hover = self._execute_hover("left sidebar")

        if "success" not in hover:
            return hover

        time.sleep(1)

        for _ in range(15):
            screenshot = self.browser.take_screenshot()
            coords = self._find_by_ocr_text(target, screenshot)

            if coords:
                x, y = coords
                self._click_at_coordinates(x, y)
                return "success"

            self._execute_scroll("down", "sidebar")

        return f"failed: {target} not found"