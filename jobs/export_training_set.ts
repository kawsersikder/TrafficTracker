import { pool } from '../backend/db';
import * as fs from 'fs';
import * as path from 'path';

async function exportTrainingSet(outputPath: string) {
  console.log(`Exporting training set to ${outputPath}...`);
  
  // Note: we'd usually join with intersection_arms to get arm_label, 
  // but right now our traffic_observations table in the schema only has arm_id.
  // Wait, in our DB schema we had:
  // traffic_observations (id, intersection_id, arm_id, observed_at, dominant_color, green_pct, ...)
  // For ML, we flatten this out. We will just dump traffic_observations.
  
  const query = `
    SELECT 
      o.id,
      o.intersection_id,
      a.arm_label,
      o.observed_at,
      o.dominant_color,
      o.green_pct,
      o.yellow_pct,
      o.red_pct,
      o.dark_red_pct,
      o.congestion_score,
      o.estimated_lane_groups,
      o.confidence
    FROM traffic_observations o
    LEFT JOIN intersection_arms a ON o.arm_id = a.id
    ORDER BY o.observed_at DESC
  `;
  
  try {
    const res = await pool.query(query);
    
    if (res.rows.length === 0) {
      console.log('No data to export.');
      return;
    }
    
    // Convert to CSV
    const headers = Object.keys(res.rows[0]).join(',');
    const rows = res.rows.map(row => {
      return Object.values(row).map(val => {
        if (val instanceof Date) return val.toISOString();
        if (typeof val === 'string') return `"${val}"`;
        return val;
      }).join(',');
    });
    
    const csvContent = [headers, ...rows].join('\n');
    fs.writeFileSync(outputPath, csvContent, 'utf-8');
    
    console.log(`Exported ${res.rows.length} rows successfully.`);
  } catch (err) {
    console.error('Export failed:', err);
  } finally {
    await pool.end();
  }
}

if (require.main === module) {
  const defaultPath = path.join(__dirname, '..', 'training_set.csv');
  exportTrainingSet(defaultPath).then(() => process.exit(0));
}
