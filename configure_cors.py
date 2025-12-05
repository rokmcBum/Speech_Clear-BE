import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

IDENTITY_URL = "https://api-identity-infrastructure.nhncloudservice.com/v2.0/tokens"
TENANT_ID = os.getenv("NHN_TENANT_ID")
USERNAME = os.getenv("NHN_USERNAME")
PASSWORD = os.getenv("NHN_PASSWORD")
STORAGE_URL = os.getenv("NHN_STORAGE_URL")
CONTAINER = os.getenv("NHN_CONTAINER")

def get_token():
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
    return res.json()["access"]["token"]["id"]

def configure_cors():
    print("Configuring CORS for container:", CONTAINER)
    
    token = get_token()
    url = f"{STORAGE_URL}/{CONTAINER}"
    
    # CORS Headers to set
    headers = {
        "X-Auth-Token": token,
        "X-Container-Read": ".r:*",  # Make container public
        "X-Container-Meta-Access-Control-Allow-Origin": "*",  # Allow all origins (or specific like http://localhost:5173)
        "X-Container-Meta-Access-Control-Expose-Headers": "Content-Length, Content-Type, ETag, X-Trans-Id, Accept-Ranges",
        "X-Container-Meta-Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
        "X-Container-Meta-Access-Control-Allow-Headers": "Content-Type, Authorization, Range"
    }
    
    # Send POST request to update container metadata
    res = requests.post(url, headers=headers)
    
    if res.status_code in [201, 202, 204]:
        print("✅ CORS configuration successful!")
        print(f"Access-Control-Allow-Origin set to: {headers['X-Container-Meta-Access-Control-Allow-Origin']}")
    else:
        print(f"❌ Failed to configure CORS. Status Code: {res.status_code}")
        print(res.text)

if __name__ == "__main__":
    configure_cors()
