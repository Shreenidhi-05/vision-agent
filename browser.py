import base64
import io
import time
import logging
from pathlib import Path
from PIL import Image
 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
 
# Set up logging so we can see what's happening step by step
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
 
 
class BrowserController:
    """
    Manages the Chrome browser session.
    
    Analogy: Think of this as the "physical body" of the agent.
    - It can open a browser (like opening your eyes)
    - It can take a screenshot (like taking a photo of what you see)
    - It can move the mouse and click (like moving your hand)
    
    But it has NO BRAIN  it doesn't decide what to click.
    That decision comes from the Vision LLM (vision_model.py).
    """
 
    def __init__(self, headless: bool = False, window_width: int = 1280, window_height: int = 900):
        """
        Initialize and launch Chrome.
        
        Args:
            headless: If True, browser runs invisibly (no window shown).
                      Set to False while debugging so you can watch the agent work!
            window_width/height: Screen size. Larger = more visible UI elements.
        """
        self.window_width = window_width
        self.window_height = window_height
        self.driver = self._launch_chrome(headless)
        logger.info(f"Browser launched. Window: {window_width}x{window_height}")
 
    def _launch_chrome(self, headless):
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.options import Options

        options = Options()

        if headless:
            options.add_argument("--headless=new")

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=options)

        driver.set_page_load_timeout(120)
        return driver

    def navigate(self, url: str):
        """
        Open a URL in the browser.
        
        Example: agent.navigate("https://myapp.com/login")
        This is like typing a URL in the address bar and pressing Enter.
        """
        logger.info(f"Navigating to: {url}")
        self.driver.get(url)
        print("CURRENT URL:", self.driver.current_url)
        time.sleep(2)  # Wait for page to start loading
 
    def take_screenshot(self, save_path: str = None) -> Image.Image:
        """
        Capture the current browser screen as an image.
        
        This is the core of our "vision" pipeline:
        1. Ask Selenium to screenshot the browser
        2. Return it as a PIL Image object
        3. Optionally save to disk (useful for debugging  you can see what the agent saw)
        
        The screenshot is then sent to the Vision LLM for understanding.
        
        Returns:
            PIL.Image.Image  the screenshot as an image object
        """
        # Selenium captures screenshot as PNG bytes
        screenshot_bytes = self.driver.get_screenshot_as_png()
        
        # Convert bytes  PIL Image (easier to work with)
        image = Image.open(io.BytesIO(screenshot_bytes))
        
        # Optionally save to disk for debugging
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)
            logger.info(f"Screenshot saved: {save_path}")
 
        return image
 
    def screenshot_as_base64(self, max_width: int = 1024) -> str:
        """
        Take a screenshot and return it as a Base64-encoded string.
        
        WHY BASE64?
            Vision LLMs (and APIs) typically accept images as Base64 strings 
            a way of encoding binary image data as text characters.
            Think of it like translating an image into a long text code.
        
        WHY RESIZE?
            Smaller images = faster LLM processing + lower API costs.
            We resize to max_width while keeping the aspect ratio.
        
        Returns:
            str  Base64 encoded JPEG image string
        """
        image = self.take_screenshot()
        
        # Resize if wider than max_width (maintains aspect ratio)
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.LANCZOS)
 
        # Convert PIL Image  JPEG bytes  Base64 string
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        jpg_bytes = buffer.getvalue()
        base64_str = base64.b64encode(jpg_bytes).decode("utf-8")
 
        logger.info(f"Screenshot captured. Size: {image.width}x{image.height}")
        return base64_str
 
    def get_page_dimensions(self) -> dict:
        """
        Get the actual scrollable dimensions of the current page.
        
        This is used when the agent needs to scroll  we need to know
        how tall the page is to decide how far to scroll.
        """
        js_script = """
        return {
            scrollWidth: document.documentElement.scrollWidth,
            scrollHeight: document.documentElement.scrollHeight,
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight,
            scrollX: window.scrollX,
            scrollY: window.scrollY
        };
        """
        return self.driver.execute_script(js_script)
 
    def quit(self):
        """Close the browser and clean up."""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed.")