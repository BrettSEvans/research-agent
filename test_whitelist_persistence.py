#!/usr/bin/env python3
"""
Test email whitelist persistence against the live Railway app.

Usage:
    RAILWAY_URL=https://your-app.railway.app \
    ADMIN_API_KEY=your-key \
    python test_whitelist_persistence.py
"""

import os
import sys
import time
import httpx

# ── Config ────────────────────────────────────────────────────────────────────
RAILWAY_URL = os.environ.get("RAILWAY_URL", "").rstrip("/")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
TEST_EMAIL = "persistence-test@saasless-test.com"
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {"X-Admin-Key": ADMIN_API_KEY, "Content-Type": "application/json"}
BASE = f"{RAILWAY_URL}/admin/whitelist"

PASS = "✅"
FAIL = "❌"
INFO = "  →"


def check_config():
    if not RAILWAY_URL:
        print(f"{FAIL} RAILWAY_URL is not set. Export it before running:")
        print("   export RAILWAY_URL=https://your-app.railway.app")
        sys.exit(1)
    if not ADMIN_API_KEY:
        print(f"{FAIL} ADMIN_API_KEY is not set. Export it before running:")
        print("   export ADMIN_API_KEY=your-key")
        sys.exit(1)
    print(f"{INFO} Target: {RAILWAY_URL}")
    print(f"{INFO} Test email: {TEST_EMAIL}\n")


def get_whitelist() -> list[dict]:
    r = httpx.get(BASE, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def add_email(email: str) -> dict:
    r = httpx.post(BASE, headers=HEADERS, json={"value": email, "added_by": "persistence-test"}, timeout=10)
    r.raise_for_status()
    return r.json()


def delete_email(entry_id: int) -> None:
    r = httpx.delete(f"{BASE}/{entry_id}", headers=HEADERS, timeout=10)
    r.raise_for_status()


def find_email(entries: list[dict], email: str) -> dict | None:
    return next((e for e in entries if e["value"] == email), None)


def cleanup(entry_id: int | None):
    if entry_id:
        try:
            delete_email(entry_id)
            print(f"\n{INFO} Cleanup: test email removed.")
        except Exception:
            print(f"\n{INFO} Cleanup: could not remove test email (id={entry_id}) — delete it manually.")


def run():
    check_config()
    entry_id = None

    # ── Step 1: Check health ──────────────────────────────────────────────────
    print("Step 1: Health check")
    try:
        r = httpx.get(f"{RAILWAY_URL}/health", timeout=10)
        print(f"{PASS} App is up (status {r.status_code})")
    except Exception as e:
        print(f"{FAIL} App unreachable: {e}")
        sys.exit(1)

    # ── Step 2: Verify no leftover test email ─────────────────────────────────
    print("\nStep 2: Pre-flight — ensure test email not already present")
    try:
        entries = get_whitelist()
        existing = find_email(entries, TEST_EMAIL)
        if existing:
            print(f"{INFO} Found leftover test email (id={existing['id']}), removing it first...")
            delete_email(existing["id"])
        print(f"{PASS} Whitelist clean ({len(entries)} entries)")
    except Exception as e:
        print(f"{FAIL} Could not read whitelist: {e}")
        sys.exit(1)

    # ── Step 3: Add test email ────────────────────────────────────────────────
    print("\nStep 3: Add test email")
    try:
        result = add_email(TEST_EMAIL)
        entry_id = result["id"]
        print(f"{PASS} Added '{TEST_EMAIL}' (id={entry_id})")
    except Exception as e:
        print(f"{FAIL} POST failed: {e}")
        sys.exit(1)

    # ── Step 4: Verify it appears immediately ─────────────────────────────────
    print("\nStep 4: Verify email appears in list immediately")
    try:
        entries = get_whitelist()
        hit = find_email(entries, TEST_EMAIL)
        if hit:
            print(f"{PASS} Found in list immediately (id={hit['id']})")
        else:
            print(f"{FAIL} Email NOT in list right after adding — something is wrong")
            cleanup(entry_id)
            sys.exit(1)
    except Exception as e:
        print(f"{FAIL} GET failed: {e}")
        cleanup(entry_id)
        sys.exit(1)

    # ── Step 5: Simulate leaving and returning (fresh connection) ─────────────
    print("\nStep 5: Simulating 'navigate away and come back' (3 second pause + fresh connection)")
    time.sleep(3)
    try:
        # Use a brand-new httpx client to simulate a fresh page load
        with httpx.Client() as fresh_client:
            r = fresh_client.get(BASE, headers=HEADERS, timeout=10)
            r.raise_for_status()
            entries = r.json()
        hit = find_email(entries, TEST_EMAIL)
        if hit:
            print(f"{PASS} Email PERSISTS after fresh connection (id={hit['id']})")
        else:
            print(f"{FAIL} Email is GONE after fresh connection — persistence is broken")
            cleanup(entry_id)
            sys.exit(1)
    except Exception as e:
        print(f"{FAIL} Fresh GET failed: {e}")
        cleanup(entry_id)
        sys.exit(1)

    # ── Step 6: Delete and verify removal ─────────────────────────────────────
    print("\nStep 6: Delete test email and verify it's removed")
    try:
        delete_email(entry_id)
        entry_id = None
        entries = get_whitelist()
        hit = find_email(entries, TEST_EMAIL)
        if not hit:
            print(f"{PASS} Email successfully deleted and no longer in list")
        else:
            print(f"{FAIL} Email still appears after delete")
            sys.exit(1)
    except Exception as e:
        print(f"{FAIL} Delete failed: {e}")
        cleanup(entry_id)
        sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print(f"{PASS} All persistence checks passed. Whitelist is working correctly.")


if __name__ == "__main__":
    run()
