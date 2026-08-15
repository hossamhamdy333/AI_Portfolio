import sys
import os
from unittest.mock import patch
from azure.core.exceptions import ServiceRequestError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("PINECONE_API_KEY", "test")

import config
from azure_storage import upload_to_blob_storage


def test_upload_skips_gracefully_when_connection_string_unset(monkeypatch):
    monkeypatch.setattr(config.settings, "AZURE_STORAGE_CONNECTION_STRING", "")
    result = upload_to_blob_storage(b"hello world", "test.txt")
    assert result is None


def test_upload_handles_azure_errors_gracefully(monkeypatch):
    # Simulate a real Azure failure (bad credentials, network issue, etc.)
    # without making an actual network call - mocked so this test is fast
    # and deterministic in any CI environment. The endpoint should log and
    # return None rather than crash the upload request, since Pinecone
    # (not blob storage) is the source of truth for retrieval.
    monkeypatch.setattr(config.settings, "AZURE_STORAGE_CONNECTION_STRING", "fake-but-nonempty")
    with patch("azure_storage._get_container_client", side_effect=ServiceRequestError("simulated failure")):
        result = upload_to_blob_storage(b"hello world", "test.txt")
    assert result is None
