# Speech Clear Backend

음성 분석 및 발표 연습을 위한 백엔드 API 서버입니다. 음성 파일을 업로드하고 분석하여 발화 속도, 피치, 음량 등의 지표를 제공하며, 개선된 음성을 합성할 수 있습니다.

## 🚀 기술 스택

- **Framework**: FastAPI
- **Database**: PostgreSQL 15
- **ORM**: SQLAlchemy
- **Authentication**: JWT
- **Language**: Python 3.11
- **Container**: Docker & Docker Compose
- **STT**: NHN Cloud Clova Speech API (음성 인식)
- **LLM**: NCP Clova Studio (문단 분할 및 피드백 생성)
- **Audio Analysis**: librosa (음성 분석)
- **Text Segmentation**: kss (한국어 문장 분할)
- **Object Storage**: NHN Cloud Object Storage

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
  - Body: `name`, `email`, `password`, `gender` (Form data)
  
- `POST /login` - 로그인
  - Body: `email`, `password` (Form data)
  - Returns: JWT 토큰

- `GET /check-email` - 이메일 중복 검증
  - Query: `email` (string)
  - Returns: 이메일 사용 가능 여부

- `GET /me` - 현재 사용자 정보 조회 (인증 필요)
  - Header: `Authorization: Bearer <token>`

### 카테고리 관리 (`/category`)

- `POST /category/` - 카테고리 생성 (인증 필요)
  - Body: `name` (Form data)
  
- `GET /category/` - 내 카테고리 목록 조회 (인증 필요)

### 음성 분석 (`/voice`)

- `POST /voice/analyze` - 음성 파일 업로드 및 분석 (인증 필요)
  - Body: `file` (Multipart file), `name` (string), `category_id` (int, 선택적, 0이면 NULL)
  - Returns: 음성 분석 결과
    - 문장 단위 세그먼트별 정보 (part, text, start, end)
    - 세그먼트별 발화 속도, 피치, 음량 등
    - LLM 기반 피드백 및 개선 사항
  - 처리 과정:
    1. Clova Speech API로 음성을 텍스트로 변환 (word-level timestamps 포함)
    2. LLM으로 문단별(서론/본론/결론) 분할
    3. kss로 각 문단을 문장 단위로 분할
    4. Clova word timestamps를 사용하여 각 문장의 정확한 시간 계산
    5. librosa로 각 문장별 상세 음성 분석 (RMS, F0, energy, pitch, rate 등)
    6. LLM 기반 피드백 생성
  - 진행률 추적: `GET /voice/progress/{user_id}` (SSE)로 실시간 진행률 확인 가능

- `GET /voice/list` - 내 음성 목록 조회 (인증 필요)
  - Query: `category_id` (int, 선택적, 없으면 전체 조회)
  - Returns: 음성 목록 (voice_id, name, category_name, created_at 등)

- `GET /voice/{voice_id}` - 단일 음성 상세 조회 (인증 필요)
  - Returns: 
    - part별로 그룹화된 segments
    - 각 segment에 0.1초마다의 dB_list 포함
    - 각 segment에 versions 배열 포함 (재녹음 버전들)
    - 각 version에도 dB_list 및 metrics 포함
    - category_id가 NULL이면 category_name은 "모든 speeches"
  - 보안: 현재 사용자가 해당 voice의 소유자인지 검증

- `GET /voice/progress/{user_id}` - 음성 분석 진행률 조회 (SSE)
  - Returns: Server-Sent Events로 실시간 진행률 스트리밍 (0-100%)

- `POST /voice/segment/{segment_id}/re_record` - 세그먼트 재녹음 (인증 필요)
  - Body: 
    - `file` (Multipart file) - 재녹음된 오디오 파일
    - `db_list` (string, 선택적) - 0.1초 간격으로 측정된 dB 값 리스트 (JSON 문자열)
  - Returns: 재녹음된 세그먼트 버전 정보 (id, version_no, text, segment_url, feedback, dB_list, metrics)
  - 보안: 현재 사용자가 해당 voice의 소유자인지 검증
  - 참고: `db_list`는 frontend에서 계산하여 전달 (전달하지 않으면 빈 배열로 저장)

- `POST /voice/synthesize/{voice_id}/` - 음성 합성 (인증 필요)
  - Returns: 개선된 최종 음성 파일 (원본과 동일한 카테고리로 저장)
  - 보안: 현재 사용자가 해당 voice의 소유자인지 검증

## 🗄️ 데이터베이스 구조

### 주요 테이블

- **users**: 사용자 정보 (이름은 암호화되어 저장)
- **categories**: 음성 카테고리 (사용자별 관리)
- **voices**: 음성 파일 메타데이터
- **voice_segments**: 음성 세그먼트 (분석 단위, 문단별 분할 정보 포함)
  - `db_list` (JSONB): 0.1초 간격으로 측정된 dB 값 리스트
- **voice_segment_versions**: 세그먼트 버전 관리 (재녹음 버전)
  - `db_list` (JSONB): 0.1초 간격으로 측정된 dB 값 리스트

### 관계

- 각 `voice`는 하나의 `category`에 속함 (NULL 가능)
- 각 `voice`는 여러 개의 `segment`를 가짐 (문장 단위로 분할됨)
- 각 `segment`는 `part` 필드를 가짐 (서론, 본론1, 본론2, 결론 등)
- 각 `segment`는 `order_no` 필드로 순서 관리
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

1. **4단계 음성 분석 파이프라인**:
   - **1단계 (Clova Speech API)**: 음성을 텍스트로 변환
     - word-level timestamps 제공
     - 정확한 시간 정보로 세그먼트 매핑
   - **2단계 (LLM)**: 전체 대본을 문단별로 분할 (서론, 본론1, 본론2, 결론 등)
     - 논리적 구조 자동 인식
   - **3단계 (kss + Clova timestamps)**: 문단을 문장 단위로 분할
     - kss 라이브러리로 한국어 문장 분할
     - Clova word timestamps를 사용하여 각 문장의 정확한 시간 계산
     - 순차적 매칭으로 겹치지 않는 세그먼트 시간 보장
   - **4단계 (librosa)**: 각 문장별로 상세 음성 분석 수행
     - RMS (Root Mean Square) - 음량 분석
     - F0 (Fundamental Frequency) - 기본 주파수
     - Energy (에너지 통계) - 평균, 표준편차, 변동계수
     - Pitch CV (Pitch 변동계수) - 음높이 안정성
     - Rate WPM (Words Per Minute) - 발화 속도
     - Pause Ratio - 휴지 비율
     - Final Boundary Features - 문장 끝 특성 (볼륨, 피치 패턴)

2. **문장 단위 세그먼트 분할**: 
   - LLM으로 문단 분할 후 kss로 문장 단위 분할
   - 각 문장이 독립적인 세그먼트로 저장
   - Clova Speech의 word timestamps를 활용한 정확한 시간 매핑
   - 순차적 매칭 알고리즘으로 겹치지 않는 시간 보장

3. **LLM 기반 피드백 생성**: 
   - 각 세그먼트의 분석 지표를 기반으로 LLM이 피드백 생성
   - Energy CV, Pitch CV, Rate WPM, Volume Ending, Pitch Ending 등 종합 분석
   - 개선 방향 제시 및 구체적인 연습 방법 안내

4. **실시간 진행률 추적**: 
   - Server-Sent Events (SSE)를 통한 실시간 진행률 스트리밍
   - 음성 분석 과정의 각 단계별 진행률 표시

5. **음성 합성**: 개선된 세그먼트들을 합성하여 최종 음성 파일 생성
   - 원본 음성과 동일한 카테고리로 자동 분류

6. **카테고리 관리**: 사용자별로 음성을 카테고리로 분류 및 관리
   - 카테고리 없이도 저장 가능 (NULL 허용)

7. **버전 관리**: 세그먼트별로 여러 버전 관리 및 비교
   - 재녹음 시 새로운 버전으로 저장
   - 각 버전마다 독립적인 dB_list 및 metrics 저장
   - 단일 voice 조회 시 모든 버전 정보 포함

8. **Object Storage 통합**: 
   - NHN Cloud Object Storage에 원본 및 세그먼트 음성 파일 저장
   - 효율적인 파일 관리 및 접근

9. **dB_list 저장**: 
   - 각 segment와 version에 0.1초 간격으로 측정된 dB 값 리스트 저장
   - PostgreSQL JSONB 타입으로 효율적인 저장 및 조회
   - 재녹음 시 frontend에서 계산된 dB_list를 받아 저장

## 🧪 테스트 데이터

`make initdb` 실행 시 다음 테스트 데이터가 자동으로 삽입됩니다:

- 테스트 사용자 (email: `test@a.com`)
- 카테고리 예시 (프레젠테이션, 회의록 등)

