import express from 'express';
import cors from 'cors';
import * as path from 'path';
import { execSync } from 'child_process';
import { captureIntersection } from '../browser/capture';
import { insertObservation, upsertIntersection, syncArms, getAnalyses, listIntersections, getHistory } from './db';
import * as fs from 'fs';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { prisma } from './db';

// Allowed countries — must match the Prisma `Country` enum.
const COUNTRIES = ['BANGLADESH', 'INDIA', 'THAILAND', 'PHILIPPINES', 'MALAYSIA'];

// Best-effort extraction of {lat, lng} from a Google Maps URL.
// Handles the common "@lat,lng,zoom" and "!3dlat!4dlng" patterns.
function parseLatLng(url: string): { latitude: number | null; longitude: number | null } {
  if (!url) return { latitude: null, longitude: null };
  const at = url.match(/@(-?\d+\.\d+),(-?\d+\.\d+)/);
  if (at) return { latitude: parseFloat(at[1]!), longitude: parseFloat(at[2]!) };
  const lat = url.match(/!3d(-?\d+\.\d+)/);
  const lng = url.match(/!4d(-?\d+\.\d+)/);
  if (lat && lng) return { latitude: parseFloat(lat[1]!), longitude: parseFloat(lng[1]!) };
  return { latitude: null, longitude: null };
}

const app = express();
app.use(cors());
app.use(express.json());

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, '..', 'public')));

const CONFIG_PATH = path.join(__dirname, '..', 'configs', 'intersections.json');
const PUBLIC_DIR = path.join(__dirname, '..', 'public');

// Helper: read all configs
function readConfigs(): any[] {
  if (!fs.existsSync(CONFIG_PATH)) return [];
  try {
    return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
  } catch {
    return [];
  }
}

// Helper: write configs
function writeConfigs(configs: any[]) {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(configs, null, 2));
}

// ─── AUTHENTICATION ───────────────────────────────────────────────
const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) throw new Error('JWT_SECRET environment variable is not set. Add it to your .env file.');

async function initAdminUser() {
  try {
    const adminCount = await prisma.user.count({ where: { role: 'ADMIN' } });
    if (adminCount === 0) {
      const password_hash = await bcrypt.hash('password', 10);
      await prisma.user.create({
        data: {
          email: 'admin@example.com',
          password_hash,
          role: 'ADMIN'
        }
      });
      console.log('[Auth] Default admin user created: admin@example.com / password');
    }
  } catch (err) {
    console.error('[Auth] Failed to init admin user:', err);
  }
}
initAdminUser();

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Email and password required' });

    const user = await prisma.user.findUnique({ where: { email } });
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });

    const isValid = await bcrypt.compare(password, user.password_hash);
    if (!isValid) return res.status(401).json({ error: 'Invalid credentials' });

    const token = jwt.sign({ userId: user.id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '24h' });
    res.json({ token, role: user.role, email: user.email });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

const requireAuth = (req: any, res: any, next: any) => {
  const authHeader = req.headers.authorization;
  if (!authHeader) return res.status(401).json({ error: 'No token provided' });
  const token = authHeader.split(' ')[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Protect all /api endpoints except /api/auth/login
app.use('/api', (req, res, next) => {
  if (req.path.startsWith('/auth/login')) {
    return next();
  }
  requireAuth(req, res, next);
});

app.get('/api/auth/me', async (req: any, res: any) => {
  try {
    const user = await prisma.user.findUnique({
      where: { id: req.user.userId },
      select: {
        id: true,
        email: true,
        role: true,
        name: true,
        designation: true,
        department: true,
        university: true,
        photo_url: true,
        created_at: true
      }
    });
    res.json({ user });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});


// ─── POST /api/capture ────────────────────────────────────────────
// Capture a screenshot from Google Maps (no analysis)
app.post('/api/capture', async (req, res) => {
  try {
    const { url, intersection_id = 'test_01' } = req.body;

    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }

    if (!fs.existsSync(PUBLIC_DIR)) {
      fs.mkdirSync(PUBLIC_DIR, { recursive: true });
    }

    console.log(`[capture] Starting capture for ${intersection_id}: ${url}`);
    const imagePath = await captureIntersection(intersection_id, url, PUBLIC_DIR);
    console.log(`[capture] Success: ${imagePath}`);

    res.json({
      success: true,
      image: `/${path.basename(imagePath)}`
    });
  } catch (err: any) {
    console.error('[capture] Error:', err);
    res.status(500).json({ error: err.message || 'Capture failed' });
  }
});

// ─── GET /api/configs ─────────────────────────────────────────────
// List all saved intersection configs
app.get('/api/configs', (req, res) => {
  try {
    const configs = readConfigs();
    res.json(configs);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ─── GET /api/config/:id ──────────────────────────────────────────
// Get a single config by intersection_id
app.get('/api/config/:id', (req, res) => {
  try {
    const configs = readConfigs();
    const config = configs.find((c: any) => c.intersection_id === req.params.id);
    if (!config) {
      return res.status(404).json({ error: 'Config not found' });
    }
    res.json(config);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ─── POST /api/save-config ────────────────────────────────────────
// Save or update a drawn line configuration
app.post('/api/save-config', async (req, res) => {
  try {
    const { intersection_id, name, google_maps_url, country, center, arms } = req.body;

    if (!intersection_id || !arms || !Array.isArray(arms) || arms.length === 0) {
      return res.status(400).json({ error: 'intersection_id and at least one arm are required' });
    }
    if (country && !COUNTRIES.includes(country)) {
      return res.status(400).json({ error: `country must be one of: ${COUNTRIES.join(', ')}` });
    }

    const { latitude, longitude } = parseLatLng(google_maps_url || '');

    const configs = readConfigs();
    const existingIndex = configs.findIndex((c: any) => c.intersection_id === intersection_id);

    const config: any = {
      intersection_id,
      name: name || intersection_id,
      country: country || null,
      google_maps_url: google_maps_url || '',
      latitude,
      longitude,
      arms
    };
    // Persist a center only if a legacy client still sends one.
    if (center) config.center = center;

    if (existingIndex >= 0) {
      configs[existingIndex] = config;
    } else {
      configs.push(config);
    }

    writeConfigs(configs);

    // Push the intersection (incl. country + geo features) and its arm
    // geometry to the database so it can be reloaded & analyzed without redraw.
    try {
      await upsertIntersection({
        id: intersection_id,
        name: name || intersection_id,
        country: country || null,
        google_maps_url: google_maps_url || '',
        latitude,
        longitude,
        arm_count: arms.length,
        expected_type: `${arms.length}-arm`,
      });
      await syncArms(intersection_id, arms);
    } catch (dbErr) {
      console.error('[save-config] DB upsert failed:', dbErr);
      // The JSON config is still saved; surface a soft warning.
    }

    console.log(`[save-config] Saved ${intersection_id} (${country || 'no country'}) with ${arms.length} arms`);

    res.json({ success: true });
  } catch (err: any) {
    console.error('[save-config] Error:', err);
    res.status(500).json({ error: err.message || 'Save failed' });
  }
});

// ─── POST /api/analyze ────────────────────────────────────────────
// Run vision analysis using the saved line-based config
app.post('/api/analyze', async (req, res) => {
  try {
    const { intersection_id } = req.body;

    if (!intersection_id) {
      return res.status(400).json({ error: 'intersection_id is required' });
    }

    // Read the saved config
    const configs = readConfigs();
    const config = configs.find((c: any) => c.intersection_id === intersection_id);
    if (!config) {
      return res.status(404).json({ error: `No saved config for ${intersection_id}` });
    }

    const snapshotPath = path.join(PUBLIC_DIR, `${intersection_id}_snapshot.png`);

    // Real-time: re-capture a fresh screenshot from the saved Maps URL so each
    // analysis reflects current traffic — no need to re-enter the link.
    let freshCapture = false;
    if (config.google_maps_url) {
      try {
        console.log(`[analyze] Re-capturing live traffic for ${intersection_id}`);
        await captureIntersection(intersection_id, config.google_maps_url, PUBLIC_DIR);
        freshCapture = true;
      } catch (capErr) {
        console.error('[analyze] Live re-capture failed, using existing snapshot:', capErr);
      }
    }

    if (!fs.existsSync(snapshotPath)) {
      return res.status(404).json({ error: `No screenshot for ${intersection_id}. Capture first.` });
    }

    // Base64-encode the config for the Python script
    const configBase64 = Buffer.from(JSON.stringify(config)).toString('base64');
    const annotatedPath = path.join(PUBLIC_DIR, `${intersection_id}_annotated.png`);
    const scriptPath = path.join(__dirname, '..', 'vision', 'extract_traffic.py');
    const venvPython = path.join(__dirname, '..', '.venv', 'bin', 'python');

    const armCount = config.arms.length;

    console.log(`[analyze] Running vision on ${intersection_id} with ${armCount} arms`);
    const stdout = execSync(
      `${venvPython} ${scriptPath} ${intersection_id} ${snapshotPath} ${armCount}-arm ${configBase64} ${annotatedPath}`
    );
    const observationData = JSON.parse(stdout.toString().trim());

    // Sync arm geometry, then store the per-arm/direction readings in the DB.
    try {
      await syncArms(intersection_id, config.arms || []);
      await insertObservation(observationData);
    } catch (dbErr) {
      console.error('[analyze] DB insert failed:', dbErr);
      // Continue even if DB insert fails — the caller still gets results.
    }

    res.json({
      success: true,
      data: observationData,
      freshCapture,
      originalImage: `/${intersection_id}_snapshot.png`,
      annotatedImage: `/${intersection_id}_annotated.png`
    });
  } catch (err: any) {
    console.error('[analyze] Error:', err);
    res.status(500).json({ error: err.message || 'Analysis failed' });
  }
});

// ─── GET /api/analyses/:id ────────────────────────────────────────
// Previous analyses for an intersection (per-arm/direction rows, newest first)
app.get('/api/analyses/:id', async (req, res) => {
  try {
    const limit = Math.min(parseInt(String(req.query.limit || '200'), 10) || 200, 1000);
    const rows = await getAnalyses(req.params.id, limit);
    res.json(rows);
  } catch (err: any) {
    console.error('[analyses] Error:', err);
    res.status(500).json({ error: err.message || 'Failed to load analyses' });
  }
});

// ─── GET /api/intersections ───────────────────────────────────────
// All intersections (DB) with country — for country grouping & filters.
app.get('/api/intersections', async (_req, res) => {
  try {
    res.json(await listIntersections());
  } catch (err: any) {
    console.error('[intersections] Error:', err);
    res.status(500).json({ error: err.message || 'Failed to load intersections' });
  }
});

// ─── GET /api/history ─────────────────────────────────────────────
// Filtered analysis history: ?country=&intersection_id=&days=
app.get('/api/history', async (req, res) => {
  try {
    const country = req.query.country ? String(req.query.country) : undefined;
    const intersectionId = req.query.intersection_id ? String(req.query.intersection_id) : undefined;
    const days = req.query.days ? parseInt(String(req.query.days), 10) : undefined;
    const rows = await getHistory({ country, intersectionId, days: days || undefined });
    res.json(rows);
  } catch (err: any) {
    console.error('[history] Error:', err);
    res.status(500).json({ error: err.message || 'Failed to load history' });
  }
});

// ─── Legacy: POST /api/scrape (kept for backward compatibility) ───
app.post('/api/scrape', async (req, res) => {
  try {
    const { url, intersection_id = 'test_01' } = req.body;
    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }

    if (!fs.existsSync(PUBLIC_DIR)) {
      fs.mkdirSync(PUBLIC_DIR, { recursive: true });
    }

    const imagePath = await captureIntersection(intersection_id, url, PUBLIC_DIR);

    const configs = readConfigs();
    const config = configs.find((c: any) => c.intersection_id === intersection_id);
    const configBase64 = config
      ? Buffer.from(JSON.stringify(config)).toString('base64')
      : 'none';

    const annotatedPath = path.join(PUBLIC_DIR, `${intersection_id}_annotated.png`);
    const scriptPath = path.join(__dirname, '..', 'vision', 'extract_traffic.py');
    const venvPython = path.join(__dirname, '..', '.venv', 'bin', 'python');

    const stdout = execSync(
      `${venvPython} ${scriptPath} ${intersection_id} ${imagePath} 4-arm ${configBase64} ${annotatedPath}`
    );
    const observationData = JSON.parse(stdout.toString().trim());

    // Save to real database
    try {
      await insertObservation(observationData);
    } catch (dbErr) {
      console.error('[scrape] DB insert failed:', dbErr);
    }

    res.json({
      success: true,
      data: observationData,
      originalImage: `/${path.basename(imagePath)}`,
      annotatedImage: `/${path.basename(annotatedPath)}`
    });
  } catch (err: any) {
    console.error('Scrape error:', err);
    res.status(500).json({ error: err.message || 'Scrape failed' });
  }
});

// ─── WEEKLY UPDATES ───────────────────────────────────────────────
const WEEKLY_UPDATES_PATH = path.join(__dirname, '..', 'configs', 'weekly-updates.json');

function getDefaultWeeklyUpdates() {
  const updates = [];
  const start = new Date('2026-06-17'); // Week 1: Wednesday June 17, 2026
  for (let i = 0; i < 12; i++) {
    const d = new Date(start);
    d.setDate(d.getDate() + i * 7);
    updates.push({
      week: i + 1,
      date: d.toISOString().split('T')[0],
      status: 'upcoming',
      title: `Week ${i + 1}`,
      description: '',
      milestones: [],
      progress: 0,
      notes: ''
    });
  }
  return updates;
}

function readWeeklyUpdates() {
  if (!fs.existsSync(WEEKLY_UPDATES_PATH)) {
    const defaults = getDefaultWeeklyUpdates();
    fs.writeFileSync(WEEKLY_UPDATES_PATH, JSON.stringify(defaults, null, 2));
    return defaults;
  }
  try {
    return JSON.parse(fs.readFileSync(WEEKLY_UPDATES_PATH, 'utf8'));
  } catch {
    return getDefaultWeeklyUpdates();
  }
}

function writeWeeklyUpdates(updates: any[]) {
  fs.writeFileSync(WEEKLY_UPDATES_PATH, JSON.stringify(updates, null, 2));
}

app.get('/api/weekly-updates', (_req, res) => {
  try {
    res.json(readWeeklyUpdates());
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.put('/api/weekly-updates/:week', (req: any, res: any) => {
  try {
    if (req.user?.role !== 'ADMIN') {
      return res.status(403).json({ error: 'Only admins can edit weekly updates' });
    }
    const weekNum = parseInt(req.params.week, 10);
    if (isNaN(weekNum) || weekNum < 1 || weekNum > 12) {
      return res.status(400).json({ error: 'Invalid week number (1–12)' });
    }
    const updates = readWeeklyUpdates();
    const idx = updates.findIndex((u: any) => u.week === weekNum);
    if (idx === -1) return res.status(404).json({ error: 'Week not found' });
    const { title, description, status, milestones, progress, notes } = req.body;
    updates[idx] = { ...updates[idx], title, description, status, milestones, progress, notes };
    writeWeeklyUpdates(updates);
    res.json({ success: true, update: updates[idx] });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
