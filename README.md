# SkillForge — Self-Evolving Agentic AI

An AI agent that starts with zero learned skills, figures out tasks using a multi-agent pipeline, then **extracts reusable skills from its own successful executions**. Next time a similar task appears, it routes to the learned skill — faster, cheaper, and more accurate.

```
┌─────────────────────────────────────────────────────────────┐
│                     New Task Arrives                        │
│                          │                                  │
│                    ┌─────▼─────┐                            │
│                    │  ROUTER   │ embedding similarity        │
│                    └─────┬─────┘                            │
│               ┌──────────┴──────────┐                       │
│               │                     │                       │
│         No match found        Match found (>55%)            │
│               │                     │                       │
│        ┌──────▼──────┐       ┌──────▼──────┐               │
│        │   EXPLORE   │       │   EXECUTE   │               │
│        │  Full agent │       │  Run stored │               │
│        │  pipeline   │       │  skill      │               │
│        └──────┬──────┘       └─────────────┘               │
│               │                                             │
│        ┌──────▼──────┐                                     │
│        │  EXTRACT    │                                     │
│        │  Learn new  │                                     │
│        │  skill      │                                     │
│        └─────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

**Explore Mode** — When no matching skill exists, a multi-agent LangGraph pipeline (planner → researcher → formatter) solves the task from scratch using Gemini. The full execution trace is logged.

**Skill Extraction** — After a successful explore run, Gemini analyzes the execution trace and extracts a generalized, reusable skill with parameters, prompt templates, and examples.

**Execute Mode** — When a matching skill is found, the system skips planning entirely and runs the stored skill's workflow with extracted parameters. Faster and cheaper.

**Self-Improvement** — The more you use it, the more skills it learns. The skill library grows organically from actual usage.

## Tech Stack

- **Backend**: Python, FastAPI, Google Gemini API
- **Skill Routing**: sentence-transformers (all-MiniLM-L6-v2) for embedding similarity
- **Storage**: SQLite
- **Frontend**: React (terminal UI with CRT aesthetic)
- **Streaming**: Server-Sent Events (SSE) for real-time agent updates

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
python run.py
```

The API runs on `http://localhost:8000`. Check health at `/api/health`.

### Frontend

Open `skillforge-ui.html` in your browser. It connects to the backend at `localhost:8000`.

Or use the React project:

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:3000` with API proxy to the backend.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/task` | Submit a task → routes to skill or explore mode |
| GET | `/api/task/{id}/stream` | SSE stream of execution events |
| GET | `/api/skills` | List all skills with stats |
| GET | `/api/skills/{id}` | Get full skill details |
| DELETE | `/api/skills/{id}` | Delete a skill |
| GET | `/api/executions` | Execution history |
| POST | `/api/feedback` | Submit feedback on an execution |
| GET | `/api/stats` | Dashboard metrics |
| POST | `/api/extract-skill/{id}` | Manually extract skill from a past run |
| GET | `/api/health` | Health check |

## Skill Schema

Each skill contains:
- **Name & description** — what the skill does generally
- **Parameters** — typed inputs with defaults (e.g., `topic`, `depth`, `num_sources`)
- **Workflow** — ordered steps with prompt templates and parameter placeholders
- **Examples** — sample inputs/outputs for routing accuracy
- **Stats** — usage count, success rate, avg execution time, cost

## Architecture

```
backend/
  app/
    main.py              FastAPI app + all endpoints
    models.py            Skill, Execution, SSEEvent schemas
    skill_store.py       SQLite + embedding-based skill router
    default_skills.py    4 pre-built starter skills
    agent_pipeline.py    Explore + Execute mode pipelines
    skill_extractor.py   Phase 2: learns skills from execution traces
  requirements.txt
  run.py

frontend/                React project (Vite)
skillforge-ui.html       Standalone terminal UI
```

## Demo

**First run (Explore)** — "Compare Supabase vs Firebase for a side project"
- No matching skill → full planner → researcher → formatter pipeline
- ~50s, 4000+ tokens
- Skill extracted: `compare_technologies`

**Second run (Execute)** — "Compare MongoDB vs PostgreSQL for analytics"
- Matched `compare_technologies` at 62% confidence
- Skipped planning → ran stored workflow directly
- Same quality, routed automatically

## Roadmap

- [x] Phase 1: Multi-agent explore pipeline + hardcoded skills
- [x] Phase 2: Automatic skill extraction from explore runs
- [ ] Phase 3: Replace embedding router with fine-tuned Laya (System One model)
- [ ] Phase 4: Skill refinement from user feedback
- [ ] Phase 5: Web search integration (Tavily) for real-time data

## Built With

Built as a portfolio project by [Harshal Balar](https://github.com/harshalbalar) — demonstrating self-evolving agentic AI, multi-agent pipelines, skill extraction, and the explore/execute routing pattern.

## License

MIT
