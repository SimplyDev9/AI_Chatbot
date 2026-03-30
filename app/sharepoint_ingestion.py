from pathlib import Path
from app.ingest_corpus import ingest_single_file
from app.sharepoint_loader import list_files_in_folder, download_file


def ingest_from_sharepoint(site_id, folder_path):
    files = list_files_in_folder(site_id)

    temp_dir = Path("temp_sharepoint")
    temp_dir.mkdir(exist_ok=True)

    for file in files:
        if "file" not in file:
            continue

        filename = file["name"]
        download_url = file["@microsoft.graph.downloadUrl"]

        local_path = temp_dir / filename

        print(f"⬇️ Downloading {filename}")
        download_file(download_url, local_path)

        print(f"📥 Ingesting {filename}")
        ingest_single_file(local_path)

    print("✅ SharePoint ingestion completed")
    print("FILES FROM SHAREPOINT:", files)