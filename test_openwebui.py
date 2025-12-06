#!/usr/bin/env python3
"""
Test Open WebUI with Playwright - capture screenshots and network traffic
"""
import asyncio
import json
import time
from playwright.async_api import async_playwright

async def test_openwebui():
    """Test Open WebUI and capture screenshots"""

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            record_video_dir="/tmp/openwebui-test"
        )

        # Enable request/response logging
        requests = []
        responses = []

        async def log_request(request):
            requests.append({
                'url': request.url,
                'method': request.method,
                'headers': dict(request.headers),
                'post_data': request.post_data
            })

        async def log_response(response):
            responses.append({
                'url': response.url,
                'status': response.status,
                'headers': dict(response.headers)
            })

        page = await context.new_page()
        page.on('request', log_request)
        page.on('response', log_response)

        # Also log console messages
        console_logs = []
        page.on('console', lambda msg: console_logs.append({
            'type': msg.type,
            'text': msg.text
        }))

        # Log page errors
        errors = []
        page.on('pageerror', lambda exc: errors.append(str(exc)))

        print("=" * 80)
        print("STEP 1: Navigate to Open WebUI")
        print("=" * 80)

        try:
            await page.goto('http://localhost:3005', timeout=30000)
            await page.wait_for_load_state('networkidle', timeout=30000)

            # Take screenshot of initial page
            await page.screenshot(path='/tmp/openwebui-01-initial.png')
            print("Screenshot saved: /tmp/openwebui-01-initial.png")

            # Get page title
            title = await page.title()
            print(f"Page title: {title}")

            # Check if login is required
            print("\n" + "=" * 80)
            print("STEP 2: Check Authentication")
            print("=" * 80)

            login_visible = await page.locator('input[type="email"], input[type="text"][placeholder*="email"]').count() > 0
            if login_visible:
                print("Login form detected")
                await page.screenshot(path='/tmp/openwebui-02-login.png')
                print("Screenshot saved: /tmp/openwebui-02-login.png")
            else:
                print("No login form - already authenticated or signup required")

            # Check for model dropdown or chat interface
            print("\n" + "=" * 80)
            print("STEP 3: Look for Model Selector")
            print("=" * 80)

            await asyncio.sleep(2)  # Wait for UI to render
            await page.screenshot(path='/tmp/openwebui-03-models.png')
            print("Screenshot saved: /tmp/openwebui-03-models.png")

            # Try to find model selector
            model_selectors = [
                'select[name="model"]',
                'button:has-text("Model")',
                'div[role="combobox"]',
                '[data-testid="model-selector"]',
                'button:has-text("gemini")',
                'button:has-text("dr-tulu")',
            ]

            for selector in model_selectors:
                count = await page.locator(selector).count()
                if count > 0:
                    print(f"Found {count} element(s) matching: {selector}")

            # Get all text content
            body_text = await page.locator('body').inner_text()
            if 'gemini' in body_text.lower():
                print("Found 'gemini' in page text")
            if 'dr-tulu' in body_text.lower():
                print("Found 'dr-tulu' in page text")
            if 'qwen' in body_text.lower():
                print("Found 'qwen' in page text")

            # Check network requests to gateway
            print("\n" + "=" * 80)
            print("STEP 4: Check Network Requests")
            print("=" * 80)

            gateway_requests = [r for r in requests if '3001' in r['url'] or 'v1/models' in r['url']]
            print(f"Found {len(gateway_requests)} requests to gateway:")
            for req in gateway_requests:
                print(f"  {req['method']} {req['url']}")

            api_requests = [r for r in requests if '/api/' in r['url']]
            print(f"\nFound {len(api_requests)} API requests:")
            for req in api_requests[:10]:  # Show first 10
                print(f"  {req['method']} {req['url']}")

            # Check for errors
            print("\n" + "=" * 80)
            print("STEP 5: Check for Errors")
            print("=" * 80)

            if errors:
                print(f"Found {len(errors)} JavaScript errors:")
                for err in errors:
                    print(f"  {err}")
            else:
                print("No JavaScript errors detected")

            # Print some console logs
            error_logs = [log for log in console_logs if log['type'] in ['error', 'warning']]
            if error_logs:
                print(f"\nFound {len(error_logs)} console errors/warnings:")
                for log in error_logs[:10]:
                    print(f"  [{log['type']}] {log['text']}")

            # Try to access models API directly
            print("\n" + "=" * 80)
            print("STEP 6: Test API Endpoints via Browser")
            print("=" * 80)

            await page.goto('http://localhost:3005/api/models', wait_until='networkidle')
            api_content = await page.content()
            if 'gemini' in api_content:
                print("API /api/models contains 'gemini'")
            await page.screenshot(path='/tmp/openwebui-04-api-models.png')
            print("Screenshot saved: /tmp/openwebui-04-api-models.png")

            # Print response data
            print("\n" + "=" * 80)
            print("STEP 7: Network Responses Summary")
            print("=" * 80)

            models_responses = [r for r in responses if 'models' in r['url']]
            print(f"Found {len(models_responses)} responses for models endpoint")

            # Save detailed logs
            with open('/tmp/openwebui-requests.json', 'w') as f:
                json.dump(requests, f, indent=2)
            print("\nDetailed requests saved to: /tmp/openwebui-requests.json")

            with open('/tmp/openwebui-responses.json', 'w') as f:
                json.dump(responses, f, indent=2)
            print("Detailed responses saved to: /tmp/openwebui-responses.json")

            with open('/tmp/openwebui-console.json', 'w') as f:
                json.dump(console_logs, f, indent=2)
            print("Console logs saved to: /tmp/openwebui-console.json")

        except Exception as e:
            print(f"Error during test: {e}")
            await page.screenshot(path='/tmp/openwebui-error.png')
            print("Error screenshot saved: /tmp/openwebui-error.png")
            raise
        finally:
            await browser.close()

if __name__ == '__main__':
    asyncio.run(test_openwebui())
