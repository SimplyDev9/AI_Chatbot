from pathlib import Path
from app.ingest_corpus import ingest_single_file
from app.sharepoint_loader import list_files_in_folder, download_file
from app.sharepoint_tracker import is_file_updated, update_tracker
from app.knowledge_base import KnowledgeBase

kb = KnowledgeBase()


def file_exists_in_db(filename):
    try:
        results = kb.vectordb.get(where={"filename": filename})
        return len(results.get("ids", [])) > 0
    except Exception as e:
        print(f"❌ DB check error: {str(e)}")
        return False


def ingest_from_sharepoint(site_id, folder_path):
    files = list_files_in_folder(site_id)

    temp_dir = Path("temp_sharepoint")
    temp_dir.mkdir(exist_ok=True)

    for file in files:
        try:
            if "file" not in file:
                continue

            filename = file["name"]
            file_id = file["id"]
            last_modified = file["lastModifiedDateTime"]
            download_url = file["@microsoft.graph.downloadUrl"]
            web_url = file.get("webUrl")

            print(f"🧠 File ID: {file_id} | Last Modified: {last_modified}")

            # 🔥 NEW LOGIC (CRITICAL FIX)
            exists_in_db = file_exists_in_db(filename)
            updated = is_file_updated(file_id, last_modified)

            if not updated and exists_in_db:
                print(f"⏩ Skipping unchanged file: {filename}")
                continue

            print(f"♻️ Re-ingesting file: {filename}")

            local_path = temp_dir / filename

            # ✅ DOWNLOAD
            print(f"⬇️ Downloading {filename}")
            download_file(download_url, local_path)

            # ✅ METADATA
            metadata = {
                "filename": filename,
                "source_type": "sharepoint",
                "source_url": web_url
            }

            # ✅ OPTIONAL (BEST PRACTICE)
            # delete old chunks before re-ingest
            try:
                kb.vectordb.delete(where={"filename": filename})
                print(f"🧹 Deleted old chunks for {filename}")
            except Exception as e:
                print(f"⚠️ Delete warning: {str(e)}")

            # ✅ INGEST
            print(f"📥 Ingesting {filename}")
            ingest_single_file(local_path, metadata=metadata)

            # ✅ UPDATE TRACKER
            update_tracker(file_id, last_modified)

        except Exception as e:
            print(f"❌ Failed file {file.get('name')}: {str(e)}")
            continue

    print("✅ SharePoint ingestion completed")