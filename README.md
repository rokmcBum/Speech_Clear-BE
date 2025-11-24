# Speech Clear Backend

음성 분석 및 발표 연습을 위한 백엔드 API 서버입니다. 음성 파일을 업로드하고 분석하여 발화 속도, 피치, 음량 등의 지표를 제공하며, 개선된 음성을 합성할 수 있습니다.

## 🚀 기술 스택

- **Framework**: FastAPI
- **Database**: PostgreSQL 15
- **ORM**: SQLAlchemy
- **Authentication**: JWT
- **Language**: Python 3.11
- **Container**: Docker & Docker Compose
- **STT**: OpenAI Whisper (음성 인식)
- **LLM**: NCP Clova Studio (문단 분할)
- **Audio Analysis**: librosa (음성 분석)

## 🛠️ 설치 및 실행

### 1. 사전 요구사항

- Python 3.11
- Conda (또는 Miniconda)
- Docker & Docker Compose

### 2. 환경 설정

#### Conda 환경 생성 및 활성화

```bash
conda env create -f environment.yml
conda activate flowclear
```


### 3. 데이터베이스 초기화

```bash
make initdb
```

이 명령어는:
- Docker 컨테이너를 재시작하고
- `db/initdb.sql` 스크립트를 실행하여 테이블을 생성하고 테스트 데이터를 삽입합니다.

### 4. 서버 실행

```bash
make run
```

이 명령어는:
- Docker 컨테이너를 시작하고
- FastAPI 서버를 `http://0.0.0.0:8080`에서 실행합니다.

### 5. API 문서 확인

서버 실행 후 다음 주소에서 Swagger UI를 확인할 수 있습니다:

```
http://0.0.0.0:8080/docs
```

## 📚 API 엔드포인트

### 사용자 인증 (`/`)

- `POST /register` - 회원가입
  - Body: `name`, `email`, `password` (Form data)
  
- `POST /login` - 로그인
  - Body: `email`, `password` (Form data)
  - Returns: JWT 토큰

- `GET /me` - 현재 사용자 정보 조회 (인증 필요)
  - Header: `Authorization: Bearer <token>`

### 카테고리 관리 (`/category`)

- `POST /category/` - 카테고리 생성 (인증 필요)
  - Body: `name` (Form data)
  
- `GET /category/` - 내 카테고리 목록 조회 (인증 필요)

### 음성 분석 (`/voice`)

- `POST /voice/analyze` - 음성 파일 업로드 및 분석 (인증 필요)
  - Query: `category_id` (int, 필수)
  - Body: `file` (Multipart file)
  - Returns: 음성 분석 결과
    - 문단별 분할 정보 (서론, 본론1, 본론2, 결론 등)
    - 세그먼트별 발화 속도, 피치, 음량 등
    - 피드백 및 개선 사항
  - 처리 과정:
    1. Whisper로 음성을 텍스트로 변환
    2. LLM으로 문단별(서론/본론/결론) 분할
    3. librosa로 각 문단별 음성 분석

- `POST /voice/segment/{segment_id}/re_record` - 세그먼트 재녹음 (인증 필요)
  - Body: `file` (Multipart file)
  - Returns: 재녹음된 세그먼트 버전 정보
  - 보안: 현재 사용자가 해당 voice의 소유자인지 검증

- `POST /voice/synthesize/{voice_id}/` - 음성 합성 (인증 필요)
  - Returns: 개선된 최종 음성 파일 (원본과 동일한 카테고리로 저장)
  - 보안: 현재 사용자가 해당 voice의 소유자인지 검증

## 🗄️ 데이터베이스 구조

### 주요 테이블

- **users**: 사용자 정보 (이름은 암호화되어 저장)
- **categories**: 음성 카테고리 (사용자별 관리)
- **voices**: 음성 파일 메타데이터
- **voice_segments**: 음성 세그먼트 (분석 단위, 문단별 분할 정보 포함)
- **voice_segment_versions**: 세그먼트 버전 관리

### 관계

- 각 `voice`는 반드시 하나의 `category`에 속함
- 각 `voice`는 여러 개의 `segment`를 가짐 (문단별로 분할됨)
- 각 `segment`는 `part` 필드를 가짐 (서론, 본론1, 본론2, 결론 등)
- 각 `segment`는 여러 개의 `version`을 가짐

## 🛠️ Make 명령어

```bash
make up      # Docker 컨테이너 시작
make down     # Docker 컨테이너 중지
make clean    # Docker 컨테이너 및 볼륨 삭제
make logs     # 컨테이너 로그 확인
make run      # 서버 실행 (컨테이너 시작 + FastAPI 실행)
make initdb   # 데이터베이스 초기화
```

## 🔐 보안

- 사용자 이름은 Fernet 암호화를 통해 저장됩니다
- 비밀번호는 bcrypt로 해시화되어 저장됩니다
- JWT 토큰을 사용한 인증 및 인가
- 민감한 정보는 환경 변수로 관리
- 리소스 접근 제어: 사용자는 자신이 소유한 voice/segment만 수정 가능
  - 세그먼트 재녹음 시 voice 소유자 검증
  - 음성 합성 시 voice 소유자 검증

## 📝 주요 기능

1. **3단계 음성 분석 파이프라인**:
   - **1단계 (Whisper)**: 음성을 텍스트로 변환 (문장별 분리)
     - 최적화된 파라미터로 문장 분할 성능 향상
     - 한국어 문장 분할 라이브러리(kss)를 통한 후처리
   - **2단계 (LLM)**: 전체 대본을 문단별로 분할 (서론, 본론1, 본론2, 결론 등)
   - **3단계 (librosa)**: 각 문단별로 음성 분석 수행
     - 발화 속도 (WPM)
     - 피치 (Hz)
     - 음량 (dB)
     - 휴지 비율
     - 운율 점수

2. **문단별 분할**: LLM을 활용하여 발표 내용을 논리적 구조로 자동 분할
   - 서론, 본론, 결론 자동 인식
   - 각 문단별 독립적인 분석 및 피드백

3. **피드백 제공**: 분석 결과를 바탕으로 개선 사항 제안
   - 단어별 상세 피드백
   - 발음, 톤, 속도 개선 제안

4. **음성 합성**: 개선된 세그먼트들을 합성하여 최종 음성 파일 생성
   - 원본 음성과 동일한 카테고리로 자동 분류

5. **카테고리 관리**: 사용자별로 음성을 카테고리로 분류 및 관리

6. **버전 관리**: 세그먼트별로 여러 버전 관리 및 비교

## 🧪 테스트 데이터

`make initdb` 실행 시 다음 테스트 데이터가 자동으로 삽입됩니다:

- 테스트 사용자 (email: `test@a.com`)
- 카테고리 예시 (프레젠테이션, 회의록 등)

