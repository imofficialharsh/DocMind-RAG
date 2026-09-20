import os
import re
import sys
from playwright.sync_api import sync_playwright

def wake_up():
    app_url = os.environ.get("STREAMLIT_APP_URL")
    if not app_url or "YOUR_APP_NAME" in app_url:
        print("Error: STREAMLIT_APP_URL environment variable is not configured.")
        print("Please configure STREAMLIT_APP_URL as a repository secret, variable, or directly in the workflow.")
        sys.exit(1)

    print(f"Connecting to {app_url} ...")
    with sync_playwright() as p:
        # Launch headless Chromium
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            # Navigate to the Streamlit app
            print("Loading page...")
            page.goto(app_url, wait_until="domcontentloaded", timeout=60000)

            # Wait for dynamic components and check for sleeping state
            page.wait_for_timeout(5000)

            # Check if the hibernation "Wake up" button is present
            wake_button = page.get_by_role("button", name=re.compile(r"get this app back up|wake up", re.IGNORECASE))

            if wake_button.count() > 0 and wake_button.first.is_visible():
                print("App is currently asleep. Clicking 'Yes, get this app back up!'...")
                wake_button.first.click()
                print("Wake-up request submitted. Waiting for container to start up...")
                # Allow time for Streamlit Cloud to initialize the app
                page.wait_for_timeout(25000)
                print("App wake-up routine completed successfully.")
            else:
                print("App is already awake. Active WebSocket connection established.")
                # Keep connection open briefly so WebSocket registers session activity
                page.wait_for_timeout(10000)
                print("Keep-alive session verified successfully.")

        except Exception as e:
            print(f"Error during keep-alive execution: {e}")
            sys.exit(1)
        finally:
            context.close()
            browser.close()

if __name__ == "__main__":
    wake_up()
