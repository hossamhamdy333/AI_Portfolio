from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError, ResourceExistsError
from config import settings, logger


def _get_container_client():
    blob_service = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(settings.AZURE_STORAGE_CONTAINER_NAME)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass  # already exists, fine
    return container_client


def upload_to_blob_storage(file_bytes: bytes, filename: str) -> str | None:
    """
    Archive the raw file to Azure Blob Storage. Best-effort: if
    AZURE_STORAGE_CONNECTION_STRING isn't set (e.g. before your Azure
    account is wired up), this logs a warning and returns None instead of
    failing the whole upload. Pinecone remains the source of truth for
    retrieval either way, so document Q&A still works without this.
    """
    if not settings.AZURE_STORAGE_CONNECTION_STRING:
        logger.warning("AZURE_STORAGE_CONNECTION_STRING not set - skipping blob archival for %s", filename)
        return None

    try:
        container_client = _get_container_client()
        container_client.upload_blob(name=filename, data=file_bytes, overwrite=True)
        account_name = container_client.account_name
        return f"https://{account_name}.blob.core.windows.net/{settings.AZURE_STORAGE_CONTAINER_NAME}/{filename}"
    except AzureError as e:
        logger.warning("Blob archival skipped for %s: %s", filename, str(e))
        return None
