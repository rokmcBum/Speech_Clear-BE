# 변수 선언
DOCKER_COMPOSE = docker compose
PYTHON = python

# 컨테이너 띄우기
up:
	$(DOCKER_COMPOSE) up -d

# 컨테이너 내리기
down:
	$(DOCKER_COMPOSE) down

# 볼륨까지 싹 지우기
clean:
	$(DOCKER_COMPOSE) down -v

# 로그 보기
logs:
	$(DOCKER_COMPOSE) logs -f

# FastAPI 실행 (로컬 실행)
run:
	$(DOCKER_COMPOSE) down -v
	$(DOCKER_COMPOSE) up -d
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# DB 초기화 (initdb.sql 반영)
initdb:
	$(DOCKER_COMPOSE) down -v
	$(DOCKER_COMPOSE) up -d

# 패키지 설치
install:
	pip install -r requirements.txt
