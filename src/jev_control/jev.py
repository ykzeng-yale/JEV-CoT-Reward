"""Pinned HTTP adapter with process-safe budget reservation and durable cache.

One shared database is required across all workers. No automatic retries. A crash,
timeout, HTTP error or malformed response retains its conservative reservation.
No key, HTTP headers or error body is persisted. Prices verified 2026-09-27.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

import httpx

MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_MILLION = 0.042
HARD_CAP_USD = 25.0
# Over 3x the advertised 65,536-token maximum at the verified input price.
RESERVE_USD = 0.01
DEFAULT_DB = Path.home() / ".local/state/jev-cot-reward/jev.sqlite3"
DEFAULT_KEY = Path.home() / ".config/jev-cot-reward/api_key"


class BudgetExceeded(RuntimeError):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class JevClient:
    def __init__(self, db=DEFAULT_DB, stage_cap_usd=0.25, transport=None, api_key=None):
        if not math.isfinite(stage_cap_usd) or not 0 < stage_cap_usd <= HARD_CAP_USD:
            raise ValueError("Stage cap must be in (0, 25] USD")
        self.db = Path(db)
        self.db.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cap = stage_cap_usd
        self.transport = transport
        self.api_key = api_key
        with self.connect() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS requests (
              id TEXT PRIMARY KEY, cache_key TEXT NOT NULL, status TEXT NOT NULL,
              charged_usd REAL NOT NULL, input_tokens INTEGER,
              started REAL NOT NULL, elapsed REAL, response TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS active_request
              ON requests(cache_key) WHERE status IN ('reserved','ok');
            ''')
        self.db.chmod(0o600)

    def connect(self):
        return sqlite3.connect(self.db, timeout=30)

    def status(self):
        with self.connect() as con:
            rows = con.execute("SELECT status, COUNT(*), SUM(charged_usd) FROM requests GROUP BY status").fetchall()
        return {"hard_cap_usd": HARD_CAP_USD, "stage_cap_usd": self.cap,
                "accounted_usd": sum(r[2] for r in rows),
                "groups": [{"status": r[0], "attempts": r[1], "usd": r[2]} for r in rows]}

    def evaluate(self, state, questions):
        # Reject unsupported/malformed schemas before
        # reserving budget or making a paid request.
        if not isinstance(state, (str, dict, list)):
            raise ValueError("State must be a string, object, or array")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 16:
            raise ValueError("Screen adapter requires a map of 1–16 typed questions")
        for name, question in questions.items():
            if not isinstance(name, str) or not name or not isinstance(question, dict):
                raise ValueError("Question names and entries must be well-formed")
            if question.get("type") not in ("noul", "choice"):
                raise ValueError("Adapter supports Noul and Choice questions")
            if not isinstance(question.get("instructions"), (str, dict, list)):
                raise ValueError("Question instructions must be a string, object, or array")
            if question["type"] == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                    raise ValueError("Choice requires 2–255 criteria")
                if any(not isinstance(k, str) or not k or not isinstance(v, (str, dict, list)) for k, v in criteria.items()):
                    raise ValueError("Invalid Choice criteria")
            elif "criteria" in question:
                criteria = question["criteria"]
                if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                    raise ValueError("Noul criteria must map true/false to descriptions")
                if any(not isinstance(value, (str, dict, list)) for value in criteria.values()):
                    raise ValueError("Noul criterion descriptions must be structured text")
        payload = {"model": MODEL, "state": state, "questions": questions}
        raw = canonical(payload)
        # Conservative local size bound; server tokenizer and usage are authoritative.
        if len(raw.encode()) > 24000 or not questions or len(questions) > 16:
            raise ValueError("Screen adapter requires <=24k UTF-8 bytes and 1–16 questions")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        request_id = str(uuid.uuid4())
        key = self.api_key or os.environ.get("TYPESAFE_API_KEY")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT status,response FROM requests WHERE cache_key=? AND status IN ('ok','reserved')", (digest,)).fetchone()
            if row:
                if row[0] == "reserved":
                    raise RuntimeError("Identical request in flight or unresolved; inspect ledger before retry")
                return {**json.loads(row[1]), "cache_hit": True}
            if not key:
                key = DEFAULT_KEY.read_text().strip()
            if not key:
                raise RuntimeError("No TypeSafe credential configured")
            spent = con.execute("SELECT COALESCE(SUM(charged_usd),0) FROM requests").fetchone()[0]
            if spent + RESERVE_USD > min(self.cap, HARD_CAP_USD) + 1e-12:
                raise BudgetExceeded("Refusing request: cumulative budget including reservations exceeded")
            con.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?)",
                        (request_id, digest, "reserved", RESERVE_USD, None, time.time(), None, None))
        started = time.perf_counter()
        try:
            if self.transport:
                response = self.transport(payload)
            else:
                with httpx.Client(timeout=60, follow_redirects=False) as http:
                    result = http.post(ENDPOINT, json=payload, headers={"Authorization": f"Bearer {key}"})
                    # Never expose response bodies or headers in failures.
                    if result.status_code != 200:
                        raise RuntimeError(f"TypeSafe returned HTTP {result.status_code}; no automatic retry")
                    response = result.json()
            if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
                raise ValueError("Malformed response/usage; reservation retained")
            n = response["usage"].get("input_tokens")
            if type(n) is not int or n < 0:
                raise ValueError("Missing valid input-token usage; reservation retained")
            if n > RESERVE_USD * 1_000_000 / PRICE_PER_MILLION:
                # Freeze even when other fields are malformed: known usage alone
                # is sufficient to invalidate the conservative reservation.
                with self.connect() as con:
                    con.execute("UPDATE requests SET status='price_anomaly',charged_usd=? WHERE id=?", (HARD_CAP_USD, request_id))
                raise RuntimeError("Usage exceeded reservation; ledger frozen for review")
            if response.get("model") != MODEL:
                raise ValueError("Unpinned model response; reservation retained")
            answers = response.get("answers", {})
            if not isinstance(answers, dict) or set(answers) != set(questions):
                raise ValueError("Question/answer mismatch; reservation retained")
            for name, question in questions.items():
                answer = answers[name]
                if not isinstance(answer, dict) or answer.get("type") != question["type"]:
                    raise ValueError("Answer type mismatch")
                if question["type"] == "noul":
                    value = answer.get("noul")
                    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid probability")
                else:
                    probs = answer.get("probabilities")
                    if not isinstance(probs, dict) or set(probs) != set(question["criteria"]):
                        raise ValueError("Choice probability keys mismatch")
                    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
                        raise ValueError("Invalid Choice probabilities")
                    if not math.isclose(sum(probs.values()), 1., abs_tol=1e-5):
                        raise ValueError("Choice probabilities do not sum to one")
                    choice = answer.get("choice")
                    if not isinstance(choice, str) or choice not in probs or probs[choice] < max(probs.values()) - 1e-8:
                        raise ValueError("Invalid Choice selection")
                    confidence = answer.get("confidence")
                    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError("Invalid Choice confidence")
            cost = n * PRICE_PER_MILLION / 1_000_000
            record = {"response": response, "elapsed_seconds": time.perf_counter() - started,
                      "input_cost_usd": cost, "request_sha256": digest}
            with self.connect() as con:
                con.execute("UPDATE requests SET status='ok',charged_usd=?,input_tokens=?,elapsed=?,response=? WHERE id=?",
                            (cost, n, record["elapsed_seconds"], canonical(record), request_id))
            return {**record, "cache_hit": False}
        except Exception:
            with self.connect() as con:
                con.execute("UPDATE requests SET status='unresolved',elapsed=? WHERE id=? AND status='reserved'",
                            (time.perf_counter()-started, request_id))
            raise
