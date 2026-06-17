import * as fs from 'fs';
import { execSync } from 'child_process';
import * as path from 'path';

async function runBatch(configPath: string) {
  const configData = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  const runnerPath = path.join(__dirname, 'run_capture.ts');
  
  console.log(`Starting batch processing for ${configData.length} intersections...`);
  
  for (const config of configData) {
    const id = config.intersection_id;
    console.log(`-----------------------------------`);
    console.log(`Processing ${id}`);
    
    try {
      // Execute the capture script synchronously for now (can be async/staggered)
      const out = execSync(`npx tsx ${runnerPath} ${configPath} ${id}`);
      console.log(out.toString());
    } catch (err: any) {
      console.error(`Batch processing failed for ${id}:`, err.message);
      if (err.stdout) console.log(err.stdout.toString());
      if (err.stderr) console.error(err.stderr.toString());
    }
    
    // Optional delay between runs to stagger
    console.log(`Waiting 5s before next intersection...`);
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
  
  console.log(`Batch complete.`);
}

if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.length < 1) {
    console.log('Usage: npx tsx jobs/run_batch.ts <config_json_path>');
    process.exit(1);
  }
  
  runBatch(args[0]).then(() => process.exit(0)).catch(err => {
    console.error(err);
    process.exit(1);
  });
}
