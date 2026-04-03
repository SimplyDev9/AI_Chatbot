import requests
from msal import ConfidentialClientApplication
import os
from pathlib import Path
from app.ingest_corpus import ingest_single_file

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]


def get_access_token():
    app = ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )

    token_response = app.acquire_token_for_client(scopes=SCOPE)

    print("FULL TOKEN RESPONSE:", token_response)

    return token_response.get("access_token")


def list_files_in_folder(site_id):
    token = get_access_token()

    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root/children"

    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)

    data = response.json()

    print("RAW SHAREPOINT RESPONSE:", data)

    return data.get("value", [])


def download_file(download_url, save_path):
    response = requests.get(download_url)

    with open(save_path, "wb") as f:
        f.write(response.content)


def ingest_from_sharepoint(site_id, folder_path="Shared Documents"):
    files = list_files_in_folder(site_id)

    temp_dir = Path("temp_sharepoint")
    temp_dir.mkdir(exist_ok=True)

    for file in files:
        if "file" not in file:
            continue

        filename = file["name"]
        download_url = file["@microsoft.graph.downloadUrl"]
        web_url = file.get("webUrl")

        local_path = temp_dir / filename

        print(f"⬇️ Downloading {filename}")
        download_file(download_url, local_path)

        # ✅ NEW METADATA
        metadata = {
            "filename": filename,
            "source_type": "sharepoint",
            "source_url": web_url
        }

        print(f"📥 Ingesting {filename}")
        ingest_single_file(local_path, metadata=metadata)

    print("✅ SharePoint ingestion completed")