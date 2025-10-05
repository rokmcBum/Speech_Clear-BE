from app.infrastructure.storage.object_storage import get_token, upload_file, download_file

TENANT_ID = "7bc368e092404db1831674f247afff6a"
USER_ID = "openlaba03"
PASSWORD = "peter123!"

STORAGE_URL = "https://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_7bc368e092404db1831674f247afff6a"
CONTAINER = "team03"  # 콘솔에서 만든 컨테이너 이름

token = get_token(TENANT_ID, USER_ID, PASSWORD)

print("token : ", token)

# 업로드
upload_url = upload_file(STORAGE_URL, token, CONTAINER, "test.wav", "../../../voice.m4a")
print("Uploaded:", upload_url)

# 다운로드
download_file(STORAGE_URL, token, CONTAINER, "test.wav", "../../../downloaded.wav")
print("Downloaded file saved.")
