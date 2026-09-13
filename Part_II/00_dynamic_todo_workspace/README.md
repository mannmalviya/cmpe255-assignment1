# Tempo

Tempo is a complete, single-user task workspace built with Next.js, TypeScript, and SQLite. It emphasizes fast capture, calm planning, durable local persistence, keyboard efficiency, responsive design, and accessible interaction.

## Run locally

Requirements: Node.js 22 or newer and npm.

```bash
cd Part_II/00_dynamic_todo_workspace
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The SQLite database is created automatically at `data/tempo.sqlite` and receives a small first-run sample set. No external service or account is required.

For a production-mode run:

```bash
npm run build
npm run start
```

## Useful commands

```bash
npm run lint       # ESLint
npm run typecheck  # TypeScript compiler
npm test           # unit tests
npm run test:e2e   # real Chrome desktop/mobile and accessibility tests
npm run check      # lint, types, unit tests, and production build
```

The end-to-end suite starts the application on port 3100 automatically. It uses the locally installed Google Chrome, exercises the complete task lifecycle, observes browser console and runtime errors, runs axe accessibility checks, and saves viewport screenshots under `docs/screenshots`.

## Features

- Persistent create, edit, complete, reopen, delete, and undo workflows
- Priority, due dates, notes, tags, full-text client search, filters, and sorting
- Pointer and keyboard-accessible drag ordering in the unfiltered manual view
- Multi-select bulk completion, reopening, and deletion
- Today, upcoming, completed, and tag-focused views
- Desktop sidebar and mobile drawer navigation
- Light/dark themes honoring the system preference on first visit
- `N` to create a task and `/` to focus search
- Validated JSON API, transactional bulk writes, WAL-mode SQLite, and health endpoint
- Reduced-motion support, semantic labels, focus treatments, and live status messages

## Project layout

```text
src/app/              Next.js pages, styles, and Route Handlers
src/components/       Interactive task workspace, editor, and rows
src/lib/              SQLite data layer, domain types, dates, validation
tests/unit/            Pure domain tests
tests/e2e/             Chrome lifecycle, responsive, console, and axe tests
data/                  Self-contained SQLite runtime files
docs/                  Design and verification artifacts
```

See [DESIGN.md](./DESIGN.md) for the product and engineering design.

