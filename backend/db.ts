import { PrismaClient } from '@prisma/client';
import * as dotenv from 'dotenv';

dotenv.config();

const prisma = new PrismaClient();

// Add support for BigInt serialization in JSON.stringify
(BigInt.prototype as any).toJSON = function () {
  return this.toString();
};

// ─── ML feature helpers ─────────────────────────────────────────────
// Weekend differs by country, so `is_weekend` is computed per-country to
// keep it a meaningful feature. ISO weekday: 1=Mon … 7=Sun.
const WEEKEND_DAYS: Record<string, number[]> = {
  BANGLADESH: [5, 6],   // Fri, Sat
  INDIA: [6, 7],        // Sat, Sun
  THAILAND: [6, 7],
  PHILIPPINES: [6, 7],
  MALAYSIA: [6, 7],
};

// Derive temporal features from a timestamp (UTC) and country.
function temporalFeatures(date: Date, country?: string | null) {
  const isoDow = ((date.getUTCDay() + 6) % 7) + 1; // JS 0=Sun → ISO 1=Mon..7=Sun
  const weekendDays = (country && WEEKEND_DAYS[country]) || [6, 7];
  return {
    hour_of_day: date.getUTCHours(),
    day_of_week: isoDow,
    is_weekend: weekendDays.includes(isoDow),
  };
}

// Congestion score in [0,1]: green is free-flow (0), darker reds are worse.
function congestionScore(t: any): number | null {
  if (!t) return null;
  const y = Number(t.yellow_pct || 0);
  const r = Number(t.red_pct || 0);
  const dr = Number(t.dark_red_pct || 0);
  return Math.min(1, Math.round((0.4 * y + 0.85 * r + 1.0 * dr) * 1000) / 1000);
}

// Replace the stored arm geometry for an intersection with the drawn arms.
// Upserts by (intersection_id, arm_label) so re-saving updates in place, and
// returns a label → arm_id map for linking observations.
export async function syncArms(
  intersectionId: string,
  arms: Array<{ name: string; points?: any[] }> = []
): Promise<Map<string, bigint>> {
  const map = new Map<string, bigint>();
  for (let i = 0; i < arms.length; i++) {
    const arm = arms[i];
    if (!arm || !arm.name) continue;
    const row = await prisma.intersectionArm.upsert({
      where: { intersection_id_arm_label: { intersection_id: intersectionId, arm_label: arm.name } },
      update: { points: arm.points ?? undefined, arm_order: i },
      create: { intersection_id: intersectionId, arm_label: arm.name, points: arm.points ?? undefined, arm_order: i },
    });
    map.set(arm.name, row.id);
  }
  return map;
}

// Upsert an intersection (used by the save-config flow so configured
// intersections — including their country — are pushed to the database).
export async function upsertIntersection(data: {
  id: string;
  name?: string;
  country?: string | null;
  google_maps_url?: string;
  latitude?: number | null;
  longitude?: number | null;
  arm_count?: number | null;
  expected_type?: string | null;
}) {
  const country = (data.country as any) || null;
  const fields = {
    name: data.name || data.id,
    country,
    google_maps_url: data.google_maps_url ?? '',
    latitude: data.latitude ?? null,
    longitude: data.longitude ?? null,
    arm_count: data.arm_count ?? null,
    expected_type: data.expected_type ?? null,
  };
  return prisma.intersection.upsert({
    where: { id: data.id },
    update: fields,
    create: { id: data.id, ...fields },
  });
}

// Persist one analysis run: a row per arm per direction (incoming/outgoing),
// each carrying the decoded colour breakdown + congestion and the denormalized
// country/temporal features. This is the ML/DL training table.
export async function insertObservation(observationData: any) {
  try {
    // Ensure the intersection exists to satisfy the foreign key constraint
    const intersection = await prisma.intersection.upsert({
      where: { id: observationData.intersection_id },
      update: {},
      create: {
        id: observationData.intersection_id,
        name: observationData.intersection_id,
        google_maps_url: '',
        expected_type: observationData.intersection_type || null,
      }
    });

    const observedAt = new Date(observationData.observed_at || Date.now());
    const temporal = temporalFeatures(observedAt, intersection.country);

    // Map arm_label → arm_id (arms are synced separately by save-config/analyze).
    const armRows = await prisma.intersectionArm.findMany({
      where: { intersection_id: observationData.intersection_id },
      select: { id: true, arm_label: true },
    });
    const armIdByLabel = new Map(armRows.map(a => [a.arm_label, a.id]));

    const arms: any[] = Array.isArray(observationData.arms) ? observationData.arms : [];
    const rows: any[] = [];
    for (const arm of arms) {
      // One reading per arm — colours along the line, plus the end-prioritized
      // signal reading and the spatial profile (start → signal).
      const end = arm.end || {};
      const signal = arm.signal || {};
      rows.push({
        intersection_id: observationData.intersection_id,
        arm_id: armIdByLabel.get(arm.arm_label) ?? null,
        observed_at: observedAt,
        country: intersection.country,
        ...temporal,
        dominant_color: arm.dominant_color ?? null,
        green_pct: arm.green_pct ?? null,
        yellow_pct: arm.yellow_pct ?? null,
        red_pct: arm.red_pct ?? null,
        dark_red_pct: arm.dark_red_pct ?? null,
        congestion_score: arm.congestion_score ?? congestionScore(arm),
        weighted_congestion_score: arm.weighted_congestion_score ?? null,
        end_dominant_color: end.dominant_color ?? null,
        end_green_pct: end.green_pct ?? null,
        end_yellow_pct: end.yellow_pct ?? null,
        end_red_pct: end.red_pct ?? null,
        end_dark_red_pct: end.dark_red_pct ?? null,
        end_congestion_score: end.congestion_score ?? null,
        signal_dominant_color: signal.dominant_color ?? null,
        signal_congestion_score: signal.congestion_score ?? null,
        red_queue_px: arm.red_queue_px ?? null,
        red_queue_frac: arm.red_queue_frac ?? null,
        arm_length_px: arm.arm_length_px ?? null,
        profile: arm.profile ?? undefined,
        estimated_lane_groups: arm.estimated_lane_groups ?? null,
        confidence: arm.confidence ?? observationData.run_confidence ?? null,
        extractor_version: observationData.extractor_version ?? null,
      });
    }

    if (rows.length === 0) {
      // No per-arm data — still record a run-level row so the analysis is logged.
      rows.push({
        intersection_id: observationData.intersection_id,
        observed_at: observedAt,
        country: intersection.country,
        ...temporal,
        extractor_version: observationData.extractor_version ?? null,
        confidence: observationData.run_confidence ?? null,
      });
    }

    const res = await prisma.trafficObservation.createMany({ data: rows });
    return res;
  } catch (err) {
    console.error('Error inserting observation via Prisma', err);
    throw err;
  }
}

// Recent analyses for an intersection (flat rows, newest first). The frontend
// groups them by observed_at into runs for the "previous analyses" table.
export async function getAnalyses(intersectionId: string, limit = 200) {
  const rows = await prisma.trafficObservation.findMany({
    where: { intersection_id: intersectionId },
    orderBy: { observed_at: 'desc' },
    take: limit,
    select: {
      observed_at: true,
      dominant_color: true,
      congestion_score: true,
      weighted_congestion_score: true,
      end_dominant_color: true,
      end_congestion_score: true,
      signal_dominant_color: true,
      signal_congestion_score: true,
      red_queue_frac: true,
      red_queue_px: true,
      profile: true,
      confidence: true,
      arm: { select: { arm_label: true } },
    },
  });
  return rows.map(r => ({
    observed_at: r.observed_at,
    arm_label: r.arm?.arm_label ?? null,
    dominant_color: r.dominant_color,
    congestion_score: r.congestion_score,
    weighted_congestion_score: r.weighted_congestion_score,
    end_dominant_color: r.end_dominant_color,
    end_congestion_score: r.end_congestion_score,
    signal_dominant_color: r.signal_dominant_color,
    signal_congestion_score: r.signal_congestion_score,
    red_queue_frac: r.red_queue_frac,
    red_queue_px: r.red_queue_px,
    profile: r.profile,
    confidence: r.confidence,
  }));
}

// All intersections (DB is the source of truth for metadata), for the team's
// country grouping and the history filters.
export async function listIntersections() {
  return prisma.intersection.findMany({
    select: { id: true, name: true, country: true, arm_count: true },
    orderBy: [{ country: 'asc' }, { name: 'asc' }],
  });
}

// Filtered analysis history across intersections. `country` is filtered on the
// denormalized observation column (no join needed); `days` limits the window.
export async function getHistory(opts: {
  country?: string; intersectionId?: string; days?: number; limit?: number;
}) {
  // Only real per-arm readings (skip empty/legacy run-level rows).
  const where: any = { dominant_color: { not: null } };
  if (opts.intersectionId) where.intersection_id = opts.intersectionId;
  if (opts.country) where.country = opts.country as any;
  if (opts.days && opts.days > 0) {
    where.observed_at = { gte: new Date(Date.now() - opts.days * 86400000) };
  }
  const rows = await prisma.trafficObservation.findMany({
    where,
    orderBy: { observed_at: 'desc' },
    take: Math.min(opts.limit ?? 1000, 5000),
    select: {
      observed_at: true,
      intersection_id: true,
      country: true,
      dominant_color: true,
      congestion_score: true,
      end_congestion_score: true,
      signal_dominant_color: true,
      signal_congestion_score: true,
      red_queue_frac: true,
      arm: { select: { arm_label: true } },
      intersection: { select: { name: true } },
    },
  });
  return rows.map(r => ({
    observed_at: r.observed_at,
    intersection_id: r.intersection_id,
    intersection_name: r.intersection?.name ?? r.intersection_id,
    country: r.country,
    arm_label: r.arm?.arm_label ?? null,
    dominant_color: r.dominant_color,
    congestion_score: r.congestion_score,
    end_congestion_score: r.end_congestion_score,
    signal_dominant_color: r.signal_dominant_color,
    signal_congestion_score: r.signal_congestion_score,
    red_queue_frac: r.red_queue_frac,
  }));
}

export async function recordProcessingRun(runData: any) {
  try {
    // Ensure the intersection exists
    await prisma.intersection.upsert({
      where: { id: runData.intersection_id },
      update: {},
      create: {
        id: runData.intersection_id,
        name: runData.intersection_id,
        google_maps_url: '',
      }
    });

    const res = await prisma.processingRun.create({
      data: {
        intersection_id: runData.intersection_id,
        started_at: new Date(runData.started_at || Date.now()),
        status: runData.status || 'STARTED',
        message: runData.message,
        extractor_version: runData.extractor_version,
      }
    });
    return res;
  } catch (err) {
    console.error('Error recording processing run via Prisma', err);
    throw err;
  }
}

export { prisma };
