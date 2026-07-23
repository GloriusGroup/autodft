# Upgrading a running controller to user accounts

This release adds accounts and per-user project namespaces. Almost all of
it is additive, but one step **rewrites existing rows in place**: every
project name changes from `X` to `admin/X`, in `molecules.project_name`
and inside the metadata of every queued entrypoint. Export directories
move from `export_data/X` to `export_data/admin/X`.

Raw `comp_data` is not touched. A failed migration therefore costs the
database file and the export tree — not the calculations.

## Before

1. **Stop the controller.** The migration runs inside `init_db()`, before
   the API starts, and must not race the worker.
2. **Snapshot the database.** `.backup` rather than `cp`: the file is in
   WAL mode and a plain copy can be torn.

   ```bash
   sqlite3 /path/to/autodft_data/autodft.db \
       ".backup '/path/to/autodft.db.pre-accounts'"
   ```

3. **Rehearse it on the snapshot.** The migration is the same code either
   way, so a clean run here is a real prediction of the live one:

   ```bash
   python - <<'EOF'
   from autodft.config import Settings
   from autodft.db import init_db
   s = Settings(); s.storage.data_path = "/path/to/a/scratch/copy"
   init_db(s)
   EOF
   ```

   Measured on a copy of this deployment's database — 135 molecules and
   1931 queued entrypoints — the first boot took 6.1 s and the second
   0.04 s. Time scales with the queued entrypoints, whose metadata JSON is
   rewritten one row at a time.

## The upgrade

Start the controller. On the first boot it will:

1. create the `users` and `projects` tables
2. create the `admin` account and **log its API key once**, in a banner.
   That key exists nowhere else — it is stored only as a hash. Copy it out
   of the log immediately. If you miss it, run
   `autodft admin rotate-key admin` on the controller.
3. give every saved header that has no owner — the six seeded methods and
   anything created before accounts existed — to `admin`
4. move every project into `admin/`, rewriting molecules and queued
   entrypoints
5. move `export_data/X` to `export_data/admin/X`

Steps 2–5 are idempotent. A second boot does nothing, and a run
interrupted partway resumes cleanly, because a name that is already
qualified is skipped.

## After

Nothing that was working stops working:

* submit scripts still send a bare project name; it is qualified with the
  caller's namespace
* the CLI gains `--user`, defaulting to `admin`

What changes for you:

* project URLs are `owner:project` — `/api/projects/admin:phenols`
* `GET /api/projects` returns qualified names
* exports live one directory deeper —
  `export_data/admin/phenols/phenols.csv`
* the CLI's *reading* commands go straight to the database and have no
  account behind them, so `--project` there wants the qualified name:
  `autodft admin export --project admin/phenols`,
  `autodft status molecules --project admin/phenols`,
  `autodft admin progress --project admin/phenols`,
  `autodft admin requeue-failed --project admin/phenols`. The same holds
  for `PipelineExtractor("admin/phenols")` in your own scripts.
  `autodft submit` is the exception: it takes a bare name plus `--user`,
  and qualifies the two itself.

Create the other accounts from the dashboard's admin page. Each person
gets their key once; hand it over out of band. To give someone their
existing work, use **reassign** rather than asking them to resubmit —
it rewrites every reference in one transaction.

## Rolling back

Stop the controller, restore the snapshot, and check out the previous
release. The only thing the old code cannot read is a project name
containing `/`, which is exactly what the snapshot predates. Exports that
were moved stay one level deep; move them back with

```bash
mv export_data/admin/* export_data/ && rmdir export_data/admin
```

## A note on what this does and does not protect

Usernames plus API keys give a real per-user boundary over the API and
the dashboard. They do not isolate the *filesystem*: everything still runs
as one Unix user, and anyone with shell access to the data directory can
read every project's `comp_data`. This is an access-control layer for the
web service, not a multi-tenant sandbox.
