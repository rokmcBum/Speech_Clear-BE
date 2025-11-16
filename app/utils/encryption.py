from cryptography.fernet import Fernet
import os

# 🚀 환경변수에서 키를 불러오거나, 없으면 새로 생성
# (처음 실행 시 자동 생성하여 .env 등에 저장해두세요)
FERNET_KEY_PATH = ".fernet.key"

if os.path.exists(FERNET_KEY_PATH):
    with open(FERNET_KEY_PATH, "rb") as f:
        SECRET_KEY = f.read()
else:
    SECRET_KEY = Fernet.generate_key()
    with open(FERNET_KEY_PATH, "wb") as f:
        f.write(SECRET_KEY)
f = Fernet(SECRET_KEY)


def encrypt_text(text: str) -> str:
    return f.encrypt(text.encode()).decode()


def decrypt_text(token: str) -> str:
    return f.decrypt(token.encode()).decode()
