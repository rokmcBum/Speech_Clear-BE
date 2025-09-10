Conda 환경 생성 (python 버전 3.11 사용)
```
conda create -n flowclear python=3.11
```

Conda 환경 활성화
```
conda activate flowclear
```

필요한 라이브러리 설치

```
pip install -U openai-whisper
pip install librosa numpy   
```

demo 파일 실행
```
python demo.py
```