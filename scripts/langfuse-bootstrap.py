#!/usr/bin/env python3
"""
ShieldPoint Langfuse Bootstrap
==============================

Runs the post-deploy setup steps for the self-hosted Langfuse stack:

  1. Wait for Langfuse health endpoint to return 200.
  2. Create the "ShieldPoint Claims Automation" project (idempotent — uses
     existing project if name matches).
  3. Create a project API key pair (pk-lf-... + sk-lf-...) and print them
     so the operator can paste into .env.
  4. Set the trace retention policy to 90 days (SHLD-9 AC).

This script uses the Langfuse v3 REST API with admin basic auth
(EMAIL/PASSWORD of the first admin user — set in the Langfuse UI on first
visit, OR use the master admin credentials you set up).

Usage:
    python3 scripts/langfuse-bootstrap.py \\
        --email admin@shieldpoint.local \\
        --password 'REPLACE_WITH_ADMIN_PASSWORD' \\
        --project-name "ShieldPoint Claims Automation" \\
        --retention-days 90

After this script runs, copy the printed API keys into .env:

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(REPO_ROOT / ".env")


class LangfuseClient:
    """Thin REST client for Langfuse v3 admin API."""

    def __init__(self, host: str, email: str, password: str) -> None:
        self.host = host.rstrip("/")
        self.email = email
        self.password = password
        self.client = httpx.Client(base_url=self.host, timeout=30.0)
        self._login()

    def _login(self) -> None:
        """Authenticate and store session cookie."""
        resp = self.client.post(
            "/api/public/auth/login",
            json={"email": self.email, "password": self.password},
        )
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(
                f"Login failed (HTTP {resp.status_code}). "
                f"Did you create the admin user in the Langfuse UI yet? "
                f"Open {self.host} in a browser, create your admin user, "
                f"then re-run this script. Body: {resp.text[:300]}"
            )

    def wait_healthy(self, max_seconds: int = 120) -> bool:
        """Poll /api/public/health until 200 or timeout."""
        print(f"[bootstrap] Waiting for {self.host}/api/public/health ...")
        start = time.time()
        while time.time() - start < max_seconds:
            try:
                resp = self.client.get("/api/public/health")
                if resp.status_code == 200:
                    print(f"[bootstrap] Langfuse healthy ({resp.status_code}).")
                    return True
            except Exception:
                pass
            time.sleep(2)
        print(f"[bootstrap] Langfuse not healthy after {max_seconds}s.", file=sys.stderr)
        return False

    def list_projects(self) -> list:
        resp = self.client.get("/api/public/projects")
        if resp.status_code != 200:
            raise RuntimeError(f"List projects failed: {resp.status_code} {resp.text[:300]}")
        return resp.json().get("data", []) or []

    def find_project_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for p in self.list_projects():
            if p.get("name") == name:
                return p
        return None

    def create_project(self, name: str) -> Dict[str, Any]:
        resp = self.client.post("/api/public/projects", json={"name": name})
        if resp.status_code in (200, 201):
            return resp.json()
        # Idempotent: if project exists, find and return it
        if resp.status_code in (400, 409):
            existing = self.find_project_by_name(name)
            if existing:
                return existing
        raise RuntimeError(
            f"Create project failed: {resp.status_code} {resp.text[:300]}"
        )

    def list_api_keys(self, project_id: str) -> list:
        resp = self.client.get(f"/api/public/projects/{project_id}/api-keys")
        if resp.status_code != 200:
            return []
        return resp.json().get("data", []) or []

    def create_api_key(self, project_id: str, note: str = "shieldpoint-agent-framework") -> Dict[str, Any]:
        resp = self.client.post(
            f"/api/public/projects/{project_id}/api-keys",
            json={"note": note},
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Create API key failed: {resp.status_code} {resp.text[:300]}"
            )
        return resp.json()

    def set_retention(self, project_id: str, days: int) -> None:
        """Set trace retention policy.

        Langfuse v3 exposes retention as a project setting. We attempt via
        the project settings endpoint; if unavailable, we record the policy
        as project metadata for visibility.
        """
        # Try the dedicated retention endpoint (v3.x)
        resp = self.client.patch(
            f"/api/public/projects/{project_id}",
            json={
                "retentionDays": days,
                "metadata": {
                    "shieldpoint.retention_days": days,
                    "shieldpoint.policy": "auto-delete-after-N-days",
                    "shieldpoint.set_by": "langfuse-bootstrap.py",
                },
            },
        )
        if resp.status_code in (200, 204):
            print(f"[bootstrap] Retention set to {days} days for project {project_id}.")
        else:
            # Fallback: just write to metadata
            print(
                f"[bootstrap] Warning: could not set retention via API "
                f"(HTTP {resp.status_code}). Manual step: open {self.host} → "
                f"Project Settings → Data Retention → set to {days} days."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="ShieldPoint Langfuse bootstrap")
    parser.add_argument(
        "--host",
        default=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        help="Langfuse base URL (default: $LANGFUSE_HOST or http://localhost:3000)",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("LANGFUSE_BOOTSTRAP_EMAIL", "admin@shieldpoint.local"),
        help="Admin user email (created via the Langfuse UI on first visit)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("LANGFUSE_BOOTSTRAP_PASSWORD"),
        help="Admin user password (will prompt if not provided)",
    )
    parser.add_argument(
        "--project-name",
        default="ShieldPoint Claims Automation",
        help="Langfuse project name (created if missing)",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("LANGFUSE_RETENTION_DAYS", "90")),
        help="Trace retention in days (default: 90, per SHLD-9 AC)",
    )
    parser.add_argument(
        "--api-key-note",
        default="shieldpoint-agent-framework",
        help="Note attached to the generated API key",
    )
    args = parser.parse_args()

    password = args.password
    if not password:
        password = getpass.getpass(f"Password for {args.email}: ")

    print(f"\n[bootstrap] ShieldPoint Langfuse Bootstrap")
    print(f"[bootstrap]   Host:           {args.host}")
    print(f"[bootstrap]   Email:          {args.email}")
    print(f"[bootstrap]   Project name:   {args.project_name}")
    print(f"[bootstrap]   Retention:      {args.retention_days} days\n")

    client = LangfuseClient(host=args.host, email=args.email, password=password)

    if not client.wait_healthy():
        return 1

    # ---- Create project (idempotent) --------------------------------------
    project = client.find_project_by_name(args.project_name)
    if project:
        print(f"[bootstrap] Project '{args.project_name}' already exists (id={project.get('id')}).")
    else:
        print(f"[bootstrap] Creating project '{args.project_name}' ...")
        project = client.create_project(args.project_name)
        print(f"[bootstrap] Project created (id={project.get('id')}).")

    project_id = project["id"]

    # ---- Set retention policy ---------------------------------------------
    client.set_retention(project_id, args.retention_days)

    # ---- Create API key (idempotent: reuse if note matches) ---------------
    existing_keys = client.list_api_keys(project_id)
    existing = next(
        (k for k in existing_keys if k.get("note") == args.api_key_note),
        None,
    )
    if existing:
        print(
            f"[bootstrap] API key with note '{args.api_key_note}' already exists. "
            "Reusing existing key — note: the SECRET value cannot be retrieved "
            "later, only the public key. If you lost the secret, delete this "
            "key in the UI and re-run."
        )
        print(f"\n[bootstrap] Public key: {existing.get('publicKey')}")
        print(f"[bootstrap] Secret key: <not retrievable — check your .env>")
    else:
        print(f"[bootstrap] Creating API key (note='{args.api_key_note}') ...")
        key = client.create_api_key(project_id, note=args.api_key_note)
        public_key = key.get("publicKey") or key.get("public_key")
        secret_key = key.get("secretKey") or key.get("secret_key")
        print(f"\n[bootstrap] ===== API KEYS — copy to .env =====")
        print(f"LANGFUSE_PUBLIC_KEY={public_key}")
        print(f"LANGFUSE_SECRET_KEY={secret_key}")
        print(f"[bootstrap] ============================================\n")

    print("[bootstrap] Done. Next steps:")
    print(f"  1. Paste the keys above into .env (if newly created)")
    print(f"  2. Verify the agent framework can send traces:")
    print(f"     python3 scripts/test-langfuse-trace.py --no-llm")
    print(f"  3. View traces in the UI: {args.host}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
