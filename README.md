# crewai-mutability-test

A minimal CrewAI crew for checking live session updates on DigitalOcean
Managed Agents: env and secret changes, a model switch and a tools change.

The platform's mediator imports `mutability_crew/crew.py` once, at start, and
runs one kickoff per turn with the turn's text as the `message` input. A live
update restarts the mediator on the new settings, so every reply shows which
process answered and with which settings.

CrewAI sessions are single-turn: each turn is its own kickoff, so there is no
conversation memory to keep across a restart.

## What it does

- **`/env`** or **`printenv`** (no model call), optionally with a name prefix
  (`/env TEST_`): the mediator's environment, as the process started with it.
  Values of names containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`,
  `CREDENTIAL` or `AUTH` are shown as `<redacted len=N sha256=XXXXXXXX>`, so a
  rotation is visible without the value. Use test values only: a short value
  can be guessed from 8 hex digits.
- **`/tools`** (no model call): the tools the `do.actions` Action Gateway
  advertises, listed with an MCP client on `HARNESS_MCP_URL_DO_ACTIONS`, plus
  that URL. Says so when `do.actions` is not attached.
- **Anything else:** with `HARNESS_INFERENCE_*` set, a one-agent Crew answers.
  Otherwise the crew echoes the message.

Every reply ends with two lines:

```
currently I'm using model: <name> (reply from: <name>)
mediator pid: <pid>, kickoff <n> in this process
```

The first name is the `HARNESS_INFERENCE_MODEL` the mediator started with.
`(reply from: …)` appears on a model reply and is the model the inference
endpoint says answered. After a restart the pid changes and the kickoff count
starts again at 1.

## Test: a live env or secret update

A `PATCH /v2/agents/sessions/{id}` with `env` or `secrets` restarts the
mediator (the whole workload nest) and then OHR, on the new environment.

1. Create a session from `agents.yaml` (set `repo_sha` first; see below).
2. Send `/env TEST_`: nothing is listed yet. Note the pid and the kickoff count.
3. PATCH the session: `env` with `TEST_FLAG: hello`, `secrets` with
   `TEST_SECRET: s3cret-v1`. Keep every existing key, since a section sent
   replaces the whole section.
4. Wait for the session to leave `UPDATING`; `session.update.state` should be
   `APPLIED_RESTARTED`.
5. Send `/env TEST_`.

**Update reached the crew:** `TEST_FLAG=hello` and `TEST_SECRET=<redacted
len=9 …>` are listed, the pid has changed and the kickoff count is 1.

Rotate the secret with a second PATCH (`TEST_SECRET: s3cret-v2`): the
`sha256=` changes and the length stays 9.

## Test: a live model switch

1. Create a session with a model and send `which model are you?`. The reply
   ends with `currently I'm using model: <old> (reply from: <old>)`.
2. PATCH the session's `env` with a new `HARNESS_INFERENCE_MODEL`, keeping the
   other keys.
3. Wait for `session.update.state` `APPLIED_RESTARTED`.
4. Send `which model are you?` again.

**Switched:** the reply ends with `currently I'm using model: <new> (reply
from: <new>)`, with a new pid and kickoff 1. The turn's `run.usage_recorded`
event should name the new model.

## Test: a live tools update (`do.actions`)

1. Create a session with `tools: [{do.actions: [web_search]}]` (see the
   commented block in `agents.yaml`).
2. Send `/tools`: the gateway's meta-tools plus `web_search`. Note the
   `do_actions url` (`…/version/<n>`) and the pid.
3. PATCH the session with `tools:\n  - do.actions: [web_search, execute_code]\n`
   as its `manifest_yaml`, as the user who created the session.
4. Wait for `session.update.state` `APPLIED_RESTARTED`.
5. Send `/tools`.

**Update reached the crew:** `execute_code` is listed, the URL ends in a new
`/version/<n+1>`, and the pid changed.

## Deploy

```bash
git rev-parse HEAD          # put this in agents.yaml as repo_sha, commit, push
doctl harness-runtime validate agents.yaml
doctl harness-runtime create agents.yaml --secret HARNESS_INFERENCE_API_KEY=@$HOME/.inference_key
```

`repo_sha` can be any pushed commit that has `crewai.json`, so pinning the
previous commit is fine.

There is no `requirements.txt`, so nothing is installed at boot: the crew uses
the crewai the `crewai` template image bakes in (1.15.11), and its `mcp`
client for `/tools`.

## Run locally

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```
