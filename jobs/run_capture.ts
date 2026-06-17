import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { captureIntersection } from '../browser/capture';
import { insertObservation, recordProcessingRun } from '../backend/db';

async function processIntersection(configPath: string, intersectionId: string) {
  const configData = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  const config = configData.find((c: any) => c.intersection_id === intersectionId);
  
  if (!config) {
    throw new Error(`Intersection ${intersectionId} not found in ${configPath}`);
  }
  
  const startTime = new Date();
  
  try {
    // 1. Browser capture
    console.log(`Starting capture for ${intersectionId}...`);
    const outputDir = path.join(__dirname, '..', 'tmp');
    const imagePath = await captureIntersection(intersectionId, config.google_maps_url, outputDir);
    
    // 2. Vision extraction
    console.log(`Extracting traffic features from ${imagePath}...`);
    const scriptPath = path.join(__dirname, '..', 'vision', 'extract_traffic.py');
    const venvPython = path.join(__dirname, '..', '.venv', 'bin', 'python');
    const stdout = execSync(`${venvPython} ${scriptPath} ${intersectionId} ${imagePath}`);
    const observationJson = stdout.toString().trim();
    const observationData = JSON.parse(observationJson);
    
    console.log('Observation extracted:', observationData);
    
    // 3. Database insert
    await insertObservation(observationData);
    
    await recordProcessingRun({
      intersection_id: intersectionId,
      started_at: startTime.toISOString(),
      finished_at: new Date().toISOString(),
      status: 'success',
      extractor_version: observationData.extractor_version
    });
    
    // Optional: cleanup image
    fs.unlinkSync(imagePath);
    console.log(`Cleaned up ${imagePath}`);
    
  } catch (error: any) {
    console.error(`Error processing ${intersectionId}:`, error);
    await recordProcessingRun({
      intersection_id: intersectionId,
      started_at: startTime.toISOString(),
      finished_at: new Date().toISOString(),
      status: 'failed',
      message: error.message || String(error)
    });
  }
}

if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.log('Usage: npx tsx jobs/run_capture.ts <config_json_path> <intersection_id>');
    process.exit(1);
  }
  
  const configPath = args[0];
  const intersectionId = args[1];
  
  processIntersection(configPath, intersectionId).then(() => {
    console.log('Done.');
    process.exit(0);
  }).catch(err => {
    console.error(err);
    process.exit(1);
  });
}
