import { MapSession } from './map-session';
import * as path from 'path';
import * as fs from 'fs';

export async function captureIntersection(intersectionId: string, url: string, outputDir: string): Promise<string> {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const outputPath = path.join(outputDir, `${intersectionId}_snapshot.png`);
  
  const session = new MapSession();
  
  try {
    // Run headed in dev mode if you want to see what it's doing, headless otherwise
    await session.init(true); 
    await session.loadMap(url);
    await session.captureScreenshot(outputPath);
    console.log(`Capture successful for ${intersectionId}`);
    return outputPath;
  } catch (error) {
    console.error(`Failed to capture ${intersectionId}:`, error);
    throw error;
  } finally {
    await session.close();
  }
}
