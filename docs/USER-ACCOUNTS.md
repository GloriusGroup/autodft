# User accounts — design

Status: **merged into `main` in 0.5.0** (commit `fb2a962`), phases 1-5
complete. This is the design record; the reference for what the API now
does is [API.md](API.md) §0, and the rollout is described in
[UPGRADE-user-accounts.md](UPGRADE-user-accounts.md).
Target: admin + per-user accounts, API keys, per-user project namespaces.

## 1. What this adds

One **admin** account keeps everything it has today: the whole API, the
whole dashboard, plus a new screen for creating users and issuing API keys.

Everyone else is a **user**. A user gets a username and one API key. With
that key they submit jobs, read results and export, over the same REST API
the admin uses — but they only ever see their own projects. On the website
they log in with username + API key and every page is filtered to them.
The `author` recorded on their submissions is their username and nobody,
admin included, can set it to anything else.

Non-goals for this branch: groups or shared projects, per-key scopes
(read-only keys), SSO/LDAP, audit log. All are additive later.

## 2. The safety guarantee

The controller runs from `/mnt/share/dft_calculations/autodft`, and since
0.4.0 that directory is what its venv's editable install resolves to.
`routes.py` imports several modules lazily *inside request handlers*, so
changing a file in that tree changes code the live controller executes on
its next request. Checking out a branch there would swap the pipeline
underneath a running 1931-molecule campaign.

So this work happens in a **git worktree** with its own **venv** and its
own **data directory**:

| | live | this branch |
|---|---|---|
| tree | `/mnt/share/dft_calculations/autodft` | `/mnt/share/dft_calculations/autodft-users` |
| venv | `…/autodft/.venv` → `autodft 0.4.1` | `…/autodft-users/.venv` → its own copy |
| data | `/mnt/share/dft_calculations/autodft_data` | a scratch directory, never the live one |
| branch | `main` | `feature/user-accounts` |

Verified: `…/autodft-users/.venv/bin/python -c "import autodft"` from `/tmp`
resolves to the worktree. Nothing on this branch reads or writes the live
database, and no migration runs against it until the branch is merged and
the controller is deliberately restarted.

(Historical, as of the merge: the branch is on `main` and the worktree is
gone. The migration now runs against the live database the first time the
controller is restarted — see
[UPGRADE-user-accounts.md](UPGRADE-user-accounts.md).)

## 3. Data model

Two new tables and one new column: `computation_headers.owner_id`, so a
saved method has an owner (see §5). **`molecules` is not altered** — `project_name` keeps
holding a string, it just holds a qualified one now, so every existing
query, join and group-by keeps working untouched.

```python
class User(table=True):          # "users"
    id, username (unique, lowercase [a-z0-9_-]{2,32})
    display_name, role: "admin" | "user"
    api_key_hash    # sha256 hex — the key itself is never stored
    api_key_prefix  # first 12 chars, so the UI can show "adft_7Kq2…"
    active, created_at, last_seen_at

class Project(table=True):       # "projects"
    id, owner_id -> users.id, name (bare), qualified_name (unique)
    created_at
    UNIQUE(owner_id, name)
```

`qualified_name` is `f"{owner.username}/{name}"` and is exactly what goes
into `molecules.project_name`. Usernames are restricted to a charset that
excludes `/`, and project names may not contain `/`, so the split is
unambiguous in both directions.

Ownership lookups are one indexed read on `projects.qualified_name`. No
`project_id` column on `molecules`: it would be a second source of truth
for something the string already answers.

## 4. Identity

```
X-AutoDFT-API-Key: adft_…      → the user who owns that key
Authorization: Bearer adft_…   → same
autodft_auth cookie            → whoever logged in
```

`X-AutoDFT-Password` was kept through 0.5.0–0.5.4 so the CLI, the
campaign scripts and saved curl invocations survived the upgrade. It was
removed in 0.5.5: a shared secret authenticates a crowd rather than a
caller, it resolved to admin — so every holder of it had the destructive
routes — and it left `project_author` saying "admin" no matter who ran
the script. Recovery from a lost key is `autodft admin rotate-key`,
which needs a shell on the controller rather than a network credential.

Keys are `adft_` + 32 url-safe random characters, shown **once** at
creation and stored only as `sha256(key)`. Lookup is by hash, so the
lookup is an index hit rather than a scan.

The session cookie is `{expires}.{username}` signed with
`Settings.session_secret()` — a random value generated once into
`<data_path>/.session_secret`. Still stateless, still invalidated
wholesale if that file is deleted, but nobody can type it: it signs, it
does not authenticate.

Resolution happens once in the auth middleware and is stashed on
`request.state.identity`; handlers take it through a `current_identity`
dependency rather than re-parsing headers.

## 5. Authorization

Three categories, and **every** `/api/*` route must be in exactly one:

* **public** — `/login`, `/logout`, static
* **admin-only** — everything under `/api/admin/*`, user management,
  database reset, circuit-breaker reset
* **scoped** — everything else: filtered to the caller's projects, or the
  whole database when the caller is admin

As built, two things ended up scoped that the first draft had as
admin-only. **Wiping** a project or a molecule moved out of
`/api/admin/*` — a user may destroy their own work, and someone else's
answers 404 exactly as a read does; only the shared `admin/default` stays
protected. **Saved headers** are shared for reading and creation and
owned for modification: anyone may list, use and create one, only its
owner or admin may edit or delete it (403, not 404 — it is already
visible in the listing).

The failure mode to design against is not a wrong check, it is a *missing*
one on a route someone adds in six months. So: a test enumerates the
routers' paths and asserts each `/api/*` one appears in one of the three
sets (`tests/test_authorization.py`). A new route with no decision
recorded fails the suite.

Reads that are not theirs return **404**, not 403 — a 403 confirms the
project exists. Writes to something that exists but is not theirs also
return 404 for the same reason.

Per the answers given: non-admins additionally get **read-only** access to
cluster status — queue depth and whether the circuit breaker has tripped —
so they can distinguish "my jobs are stuck" from "the pipeline is halted".
They cannot reset either.

## 6. Namespacing

Chosen over globally-unique names, so two people can both have
`screening`. This is the largest surface in the change: 70 `project_name`
sites across 11 modules and 125 mentions in the dashboard template.

What absorbs it: **the qualified name is a string in the same field the
bare name used to be in.** Nothing that treats a project as an opaque key
has to change. What does change:

* **Submission** — the request body keeps sending a *bare* name; the
  server qualifies it with the caller's username. Existing submit scripts
  are unaffected.
* **Routes** — a project is addressed `owner:project` in a URL. Not
  `{owner}/{name}`: that collides with `/api/projects/{name}/export` for
  any project called "export", and a percent-encoded slash is normalised
  back to a separator before routing, so `owner%2Fproject` never reaches
  the handler either. `/api/projects/{name}` still accepts a bare name,
  meaning "mine"; for admin it resolves to a unique match or answers 409
  listing the candidates. Old bookmarks and CLI invocations keep working.
* **Exports** — `export_data/{owner}/{name}`. `safe_subdirectory` gains a
  two-segment form; both segments are validated, so a crafted owner or
  project name still cannot escape the root.
* **comp_data** — unchanged. Molecule directories stay `mol_{id}`;
  ownership is reached through the molecule's project. No file moves for
  raw data, which is the bulk of the disk.

## 7. Submissions and the author field

`SubmitRequest.author` is accepted and ignored: the recorded author is
always the calling account's username, admin included. An exemption for
admin would be an exemption for whoever holds the shared dashboard
password, which is the one credential that is not personal. To record
work as someone else, submit it with their key. The dashboard form shows
the field pre-filled and read-only for everyone.

Submitting to a project name the caller does not own creates *their* copy
in their namespace rather than joining someone else's — with namespacing
that is the natural reading and it removes a whole class of accidental
cross-writes.

## 8. Admin UI

A new **Users** section, admin-only:

* create a user (username, display name, role) → shows the API key once,
  with a clear "this is the only time you will see it" warning
* rotate a key, deactivate/reactivate a user
* reassign a project to another owner
* per-user summary: projects, molecules, jobs in flight

Deleting a user is deliberately *not* one click: it must go through the
existing project-wipe flow per project, or reassignment. Removing an
account whose projects still hold hundreds of gigabytes should be an
explicit sequence of decisions.

## 9. Migration

On first boot of the new version, inside the existing `db.py` migration
mechanism:

1. create `users` and `projects`
2. create the `admin` user; generate its API key and **log it once** with
   a banner (it cannot be recovered afterwards, only rotated)
3. adopt every ownerless `ComputationHeader` — the seeded methods and
   anything predating accounts — into admin
4. for every distinct `molecules.project_name`, create a `Project` owned
   by admin and rewrite the column from `X` to `admin/X`, in `molecules`
   and inside each queued entrypoint's metadata JSON
5. move `export_data/X` to `export_data/admin/X`

Step 4 rewrites production rows and step 5 moves directories, so this is
the one genuinely irreversible part of the change. It gets: a dry-run
mode that reports what it would do, a test against a copy of a real
database, and a documented instruction to snapshot `autodft.db` first.
Raw `comp_data` is untouched, so a botched migration costs the database
file and the export tree — not the calculations.

## 10. Test strategy

* unit: key generation/hashing, username and project-name validation,
  qualified-name round-tripping, cookie signing with a username
* authorization matrix: for each of {admin, owner, other user, no
  credential} × {read, export, modify, wipe} × {own, other's project},
  assert the exact status code
* the route-coverage meta-test from §5
* migration: build a database in the pre-migration shape, migrate, assert
  every molecule reachable and owned by admin, and that a second run is a
  no-op
* regression: every pre-existing test must pass unchanged, since all of them
  exercise the admin path (the suite stood at 230 when this was written
  and at 378 once the work landed)

## 11. Phases

1. **Models and identity** — `User`, `Project`, key generation, migration,
   identity resolution. No enforcement yet; the suite stays green because
   everything still resolves to admin.
2. **Enforcement** — `current_identity`, project scoping, the route
   coverage meta-test, the authorization matrix.
3. **Namespacing** — qualified names, the new routes, export paths.
4. **UI** — login with username + key, the Users admin section, per-user
   filtering in the dashboard, the locked author field.
5. **Docs** — `docs/API.md` authentication section, README, and an upgrade
   note covering the migration.
