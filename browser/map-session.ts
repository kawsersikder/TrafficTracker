import { chromium, Browser, Page } from 'playwright';

export class MapSession {
  private browser: Browser | null = null;
  private page: Page | null = null;

  async init(headless: boolean = true) {
    this.browser = await chromium.launch({ headless });
    const context = await this.browser.newContext({
      viewport: { width: 1280, height: 800 },
      deviceScaleFactor: 1, // Ensure consistent scaling
    });
    this.page = await context.newPage();
  }

  async loadMap(url: string) {
    if (!this.page) throw new Error('Session not initialized');

    await this.page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

    // Wait for the map canvas to be loaded
    console.log('Waiting for map to render...');
    await this.page.waitForSelector('canvas', { state: 'visible', timeout: 15000 });
    
    // Additional wait for traffic layer to stabilize
    // Google Maps traffic layer can take a few seconds to fully render over the base map
    await this.page.waitForTimeout(5000); 

    // Hide UI overlays if needed
    await this.hideOverlays();
  }

  private async hideOverlays() {
    if (!this.page) return;
    
    // We try to hide common Google Maps UI elements so they don't interfere with our screenshot
    try {
      await this.page.evaluate(() => {
        const elementsToHide = [
          '#omnibox-container', // Search box
          '#vasquette', // Bottom bar
          '.app-viewcard-strip', // Side panel
          '.scene-footer-container', // Bottom right controls
          '.widget-settings-button', // Settings
          '#titlecard', // Place info panel
          'div[role="dialog"]', // Floating dialogue boxes
          '#QA0Szd', // Entire side panel and floating cards container
        ];
        elementsToHide.forEach(selector => {
          const els = document.querySelectorAll(selector);
          els.forEach(el => {
            (el as HTMLElement).style.display = 'none';
          });
        });
      });
    } catch (e) {
      console.warn('Failed to hide some UI overlays:', e);
    }
  }

  async captureScreenshot(outputPath: string) {
    if (!this.page) throw new Error('Session not initialized');
    
    console.log(`Saving screenshot to ${outputPath}...`);
    await this.page.screenshot({ path: outputPath });
  }

  async close() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
      this.page = null;
    }
  }
}
