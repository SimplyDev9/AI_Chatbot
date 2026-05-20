import os
from pathlib import Path

import requests
from msal import ConfidentialClientApplication

from app.ingest_corpus import ingest_single_file
from app.logger import logger

TENANT_ID     = os.getenv("TENANT_ID")
CLIENT_ID     = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE     = ["https://graph.microsoft.com/.default"]


def get_access_token() -> str:
    app = ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )

    token_response = app.acquire_token_for_client(scopes=SCOPE)

    # ── NEVER log the token itself — it is a bearer credential ───────────────
    # The old code did: print("FULL TOKEN RESPONSE:", token_response)
    # In containerised environments stdout goes to CloudWatch / ELK, so the
    # full OAuth token would be stored in your log aggregator indefinitely.
    token = token_response.get("access_token")
    if not token:
        error = token_response.get("error", "unknown")
        desc  = token_response.get("error_description", "no description")
        logger.error("SharePoint: failed to acquire token. error=%s desc=%s", error, desc)
        raise RuntimeError(f"SharePoint authentication failed: {error} — {desc}")

    logger.info("SharePoint: access token acquired successfully")
    return token


def list_files_in_folder(site_id: str) -> list:
    token = get_access_token()

    url     = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root/children"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    # ── Log only safe metadata, never raw payload (may contain file content) ──
    # The old code did: print("RAW SHAREPOINT RESPONSE:", data)
    files = data.get("value", [])
    logger.info("SharePoint: listed %d items from site %s", len(files), site_id)

    return files


def download_file(download_url: str, save_path: Path) -> None:
    response = requests.get(download_url)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    logger.debug("SharePoint: downloaded file to %s (%d bytes)", save_path, len(response.content))