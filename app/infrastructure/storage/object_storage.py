import requests

IDENTITY_URL = "https://api-identity-infrastructure.nhncloudservice.com/v2.0/tokens"


def get_token(tenant_id, username, password):
    payload = {
        "auth": {
            "tenantId": tenant_id,
            "passwordCredentials": {
                "username": username,
                "password": password
            }
        }
    }
    res = requests.post(IDENTITY_URL, json=payload)
    res.raise_for_status()
    token = res.json()["access"]["token"]["id"]
    return token


def upload_file(storage_url, token, container, object_name, file_path):
    url = f"{storage_url}/{container}/{object_name}"
    headers = {"X-Auth-Token": token}
    with open(file_path, "rb") as f:
        res = requests.put(url, headers=headers, data=f)
    res.raise_for_status()
    return url


def download_file(storage_url, token, container, object_name, save_path):
    url = f"{storage_url}/{container}/{object_name}"
    headers = {"X-Auth-Token": token}
    res = requests.get(url, headers=headers, stream=True)
    res.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path
