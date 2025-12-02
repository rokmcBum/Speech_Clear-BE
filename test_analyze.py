import requests

# 1. 로그인하여 토큰 발급 (테스트용 계정이 있다고 가정)
# 만약 테스트 계정이 없다면 회원가입 로직이 필요할 수 있음.
# 여기서는 하드코딩된 토큰이나, 로컬 DB에 있는 유저로 로그인 시도
# 일단 토큰 없이 401이 뜨는지, 토큰 있으면 422가 안뜨는지 확인

# 가상의 토큰 (실제 토큰이 필요함)
# 로그인 API를 먼저 호출해서 토큰을 받아오자.
BASE_URL = "http://127.0.0.1:8000"

def get_token():
    # 로그인 시도 (DB에 있는 유저 정보 필요)
    # initdb.sql에 보면 user_id=1인 유저가 있을 것임.
    # 비밀번호를 모르면 새로 가입하거나, verify_password를 우회해야 함.
    # 하지만 여기서는 사용자가 이미 실행 중인 서버에 요청을 보내는 것이므로,
    # 사용자가 제공한 로그에서 토큰을 얻을 수는 없음.
    
    # 대신, 간단히 422 에러의 원인만 파악하기 위해
    # "토큰이 유효하지 않음" (401)이 뜨면 적어도 422는 통과한 것임.
    # 422는 파라미터 검증 에러이므로, 인증보다 먼저 발생할 수도 있고 나중에 발생할 수도 있음.
    # FastAPI는 Depends(get_current_user)가 먼저 실행되므로 401이 먼저 뜰 것임.
    # 만약 401이 뜬다면 -> 파라미터 검증 단계까지 못 간 것.
    # 만약 422가 뜬다면 -> 파라미터가 잘못된 것.
    return "dummy_token"

def test_analyze():
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # 파일 생성
    files = {'file': ('test.wav', b'dummy content', 'audio/wav')}
    data = {
        'category_id': 0, # 0으로 테스트
        'name': 'test_voice'
    }
    
    print("Sending request...")
    try:
        response = requests.post(f"{BASE_URL}/voice/analyze", files=files, data=data, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_analyze()
