# app/infrastructure/storage/object_storage.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()


IDENTITY_URL = "https://api-identity-infrastructure.nhncloudservice.com/v2.0/tokens"
TENANT_ID = os.getenv("NHN_TENANT_ID")
USERNAME = os.getenv("NHN_USERNAME")
PASSWORD = os.getenv("NHN_PASSWORD")
STORAGE_URL = os.getenv("NHN_STORAGE_URL")
CONTAINER = os.getenv("NHN_CONTAINER")
_token_cache = None


def get_token():
    global _token_cache
    if _token_cache:  # TODO: 만료시간 확인 후 갱신
        return _token_cache

    payload = {
        "auth": {
            "tenantId": TENANT_ID,
            "passwordCredentials": {
                "username": USERNAME,
                "password": PASSWORD
            }
        }
    }
    res = requests.post(IDENTITY_URL, json=payload)

    res.raise_for_status()
    _token_cache = res.json()["access"]["token"]["id"]
    return _token_cache


def upload_file(local_path: str, object_name: str) -> str:
    url = f"{STORAGE_URL}/{CONTAINER}/{object_name}"
    token = get_token()
    headers = {"X-Auth-Token": token}
    with open(local_path, "rb") as f:
        res = requests.put(url, headers=headers, data=f)
    res.raise_for_status()
    return url


def download_file(object_name: str, save_path: str) -> str:
    # url = f"{STORAGE_URL}/{CONTAINER}/{object_name}"
    token = get_token()
    headers = {"X-Auth-Token": token}
    res = requests.get(object_name, headers=headers, stream=True)
    res.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path
