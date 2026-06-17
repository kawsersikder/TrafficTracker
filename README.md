# Traffic Intersection Analyzer

> **CSE498R — Computer Science Research Project**  
> North South University · Department of Electrical and Computer Engineering

An AI-powered web application that captures live Google Maps traffic screenshots, lets researchers draw road arm geometries on top of them, and runs a computer vision pipeline (OpenCV + Claude Vision) to score congestion on each arm in real time. Results are stored in PostgreSQL and browsable through a role-based dashboard.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Architecture](#architecture)
4. [Prerequisites](#prerequisites)
5. [Setup](#setup)
6. [Environment Variables](#environment-variables)
7. [Database Setup](#database-setup)
8. [Running the Server](#running-the-server)
9. [Creating Users](#creating-users)
10. [Project Structure](#project-structure)
11. [API Reference](#api-reference)
12. [Python Vision Pipeline](#python-vision-pipeline)
13. [Weekly Updates](#weekly-updates)
14. [License](#license)

---

## Features

- **Live screenshot capture** — headless Playwright browser captures Google Maps traffic layer on demand
- **Interactive road arm drawing** — canvas-based polyline editor to map each road arm's geometry
- **AI congestion analysis** — Python/OpenCV pipeline scores green/yellow/red congestion per arm, with stop-line signal detection
- **Role-based access**
  - `ADMIN` — full access: draw arms, run analysis, edit weekly updates
  - `TEACHER` — read-only dashboard: view results, reports, and weekly progress
- **Reports page** — filterable history table across all intersections, countries, and time windows
- **Weekly Updates** — 12-week research timeline with interactive progress cards (admin-editable)
- **JWT authentication** — token-based login with 24-hour sessions

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS (Claude design system) |
| Backend | Node.js + Express 5 + TypeScript (tsx) |
| Database | PostgreSQL + Prisma ORM |
| Vision | Python 3 + OpenCV + NumPy + Pydantic |
| Browser automation | Playwright (Chromium) |
| Auth | bcrypt + JSON Web Tokens |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (SPA)                       │
│  Home · Reports · Weekly Update  (role-aware nav)       │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP / REST
┌────────────────────▼────────────────────────────────────┐
│            Express Server  (backend/server.ts)          │
│  /api/auth  /api/capture  /api/analyze  /api/history   │
│  /api/configs  /api/weekly-updates                     │
└──────┬──────────────────────────────────┬───────────────┘
       │ Prisma ORM                       │ child_process
┌──────▼──────────┐             ┌─────────▼───────────────┐
│   PostgreSQL    │             │   Python Vision Pipeline  │
│  users          │             │   extract_traffic.py      │
│  intersections  │             │   OpenCV · NumPy          │
│  intersection_  │             │   Congestion scoring      │
│    arms         │             │   Annotated PNG output    │
│  traffic_       │             └─────────────────────────-┘
│    observations │
│  processing_    │
│    runs         │
└─────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Node.js | ≥ 20 |
| npm | ≥ 10 |
| Python | ≥ 3.10 |
| PostgreSQL | ≥ 14 |
| Chromium (Playwright) | installed via `npx playwright install` |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/cse498r-traffic.git
cd cse498r-traffic
```

### 2. Install Node dependencies

```bash
npm install
```

### 3. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Install Playwright browser

```bash
npx playwright install chromium
```

### 5. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials and a secure JWT secret
```

See [Environment Variables](#environment-variables) for details.

### 6. Set up the database

```bash
npx prisma generate          # generates the Prisma client
npx prisma db push           # applies the schema to your PostgreSQL database
```

### 7. Create your first user

```bash
npx tsx scripts/create-user.ts admin@example.com yourpassword ADMIN "Your Name"
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in all values:

| Variable | Description | Example |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost:5432/trafficdb` |
| `JWT_SECRET` | Secret key for signing JWTs — **use a long random string** | `openssl rand -hex 32` |
| `PORT` | HTTP port for the Express server | `3000` |

> **Never commit `.env` to version control.** It is listed in `.gitignore`.

---

## Database Setup

The schema is managed by Prisma. After setting `DATABASE_URL` in `.env`:

```bash
# Push the full schema (creates all tables)
npx prisma db push

# Or, if you prefer migrations:
npx prisma migrate dev --name init
```

### Schema overview

```
users                — accounts (ADMIN | TEACHER roles)
intersections        — saved intersection configs (name, country, GPS)
intersection_arms    — per-arm polyline geometry
traffic_observations — per-arm congestion readings (per analysis run)
processing_runs      — log of vision pipeline executions
```

The full Prisma schema is at [`prisma/schema.prisma`](prisma/schema.prisma).

### Supported countries (enum)

`BANGLADESH` · `INDIA` · `THAILAND` · `PHILIPPINES` · `MALAYSIA`

---

## Running the Server

```bash
# Development (tsx hot-ish reload)
npm run dev

# Or directly
npx tsx backend/server.ts
```

The server starts on `http://localhost:3000` (or the `PORT` in `.env`).  
It serves the frontend from `public/` and exposes all `/api/*` routes.

---

## Creating Users

Use the generic CLI script — **never hardcode credentials in source files**:

```bash
# Create a TEACHER account
npx tsx scripts/create-user.ts teacher@university.edu "SecurePass!" TEACHER "Dr. Jane Smith"

# Create an ADMIN account
npx tsx scripts/create-user.ts admin@university.edu "SecurePass!" ADMIN "John Doe"
```

Both `upsert` safely — running the script again with the same email updates the record.

---

## Project Structure

```
traffic/
├── backend/
│   ├── server.ts          # Express app — all routes & auth
│   └── db.ts              # Prisma client + DB helper functions
├── browser/
│   ├── capture.ts         # Playwright screenshot capture
│   └── map-session.ts     # Google Maps session management
├── configs/
│   ├── intersections.json # Saved intersection arm configs
│   └── weekly-updates.json# 12-week project timeline data
├── jobs/
│   ├── run_batch.ts       # Batch analysis runner
│   ├── run_capture.ts     # Batch capture runner
│   └── export_training_set.ts
├── prisma/
│   └── schema.prisma      # Database schema
├── public/
│   ├── index.html         # Main SPA (Home · Reports · Weekly Update)
│   └── login.html         # Login page
├── scripts/
│   └── create-user.ts     # Generic user creation CLI
├── vision/
│   ├── extract_traffic.py # Main vision pipeline (congestion scoring)
│   ├── preprocess.py      # Image preprocessing utilities
│   ├── schema.py          # Pydantic output schema
│   └── detect_intersection.py
├── .env.example           # Environment variable template
├── .gitignore
├── package.json
├── requirements.txt       # Python dependencies
├── tsconfig.json
└── README.md
```

---

## API Reference

All endpoints (except `/api/auth/login`) require a `Bearer <token>` header.

| Method | Path | Role | Description |
|---|---|---|---|
| `POST` | `/api/auth/login` | Public | Sign in, returns JWT |
| `GET` | `/api/auth/me` | Any | Current user profile |
| `GET` | `/api/configs` | Any | List all intersection configs |
| `GET` | `/api/config/:id` | Any | Single intersection config |
| `POST` | `/api/save-config` | Any | Create or update an intersection config |
| `POST` | `/api/capture` | Any | Capture a live Google Maps screenshot |
| `POST` | `/api/analyze` | Any | Run the vision pipeline on an intersection |
| `GET` | `/api/analyses/:id` | Any | Per-arm readings for an intersection |
| `GET` | `/api/intersections` | Any | All intersections (for filter dropdowns) |
| `GET` | `/api/history` | Any | Filtered analysis history (`?country=&intersection_id=&days=`) |
| `GET` | `/api/weekly-updates` | Any | All 12-week update cards |
| `PUT` | `/api/weekly-updates/:week` | ADMIN | Edit a week card (1–12) |

---

## Python Vision Pipeline

The vision pipeline is invoked by the backend as a subprocess:

```bash
.venv/bin/python vision/extract_traffic.py \
  <intersection_id> \
  <snapshot_path.png> \
  <N>-arm \
  <base64_config> \
  <output_annotated.png>
```

It outputs a JSON object to stdout containing per-arm congestion scores, dominant signal colours, red-queue fractions, and spatial colour profiles.

**Required packages** (`requirements.txt`):
```
opencv-python-headless>=4.8.0
numpy>=1.24.0
pydantic>=2.0.0
```

---

## Weekly Updates

The 12-week project timeline is stored in `configs/weekly-updates.json` and editable through the UI by ADMIN users. Weeks run every Wednesday starting **17 June 2026** through **2 September 2026**.

| Week | Date |
|---|---|
| 1 | Wednesday, 17 Jun 2026 |
| 2 | Wednesday, 24 Jun 2026 |
| 3 | Wednesday, 01 Jul 2026 |
| 4 | Wednesday, 08 Jul 2026 |
| 5 | Wednesday, 15 Jul 2026 |
| 6 | Wednesday, 22 Jul 2026 |
| 7 | Wednesday, 29 Jul 2026 |
| 8 | Wednesday, 05 Aug 2026 |
| 9 | Wednesday, 12 Aug 2026 |
| 10 | Wednesday, 19 Aug 2026 |
| 11 | Wednesday, 26 Aug 2026 |
| 12 | Wednesday, 02 Sep 2026 |

---

## License

This project is part of the CSE498R Research course at North South University.  
All rights reserved © 2026 — The Research Team.
