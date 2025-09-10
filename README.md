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

결과
```
{
  "text": " 안녕하세요 저는 동물매개팀의 조장 이태권입니다 반갑습니다 이번에 저희는 동물매개라는 주제를 가지고 활동을 했었는데요 활동을 하면서 간행을 겪기도 했었고부터 발표를 시작하겠습니다 감사합니다",
  "segments": [
    {
      "sid": 0,
      "text": "안녕하세요 저는 동물매개팀의 조장 이태권입니다 반갑습니다 이번에 저희는 동물매개라는 주제를 가지고 활동을 했었는데요",
      "start": 1.6799999999999993,
      "end": 11.8,
      "metrics": {
        "dB": -14.71,
        "pitch_mean_hz": 88.65,
        "pitch_std_hz": 11.88,
        "rate_wpm": 77.1,
        "pause_ratio": 0.007,
        "prosody_score": 11.79
      },
      "words": [
        {
          "text": "안녕하세요",
          "start": 1.6799999999999993,
          "end": 2.9,
          "metrics": {
            "dB": -19.78,
            "pitch_mean_hz": 81.97,
            "pitch_std_hz": 14.47,
            "duration_sec": 1.22
          }
        },
        {
          "text": "저는",
          "start": 2.9,
          "end": 3.56,
          "metrics": {
            "dB": -5.75,
            "pitch_mean_hz": 94.67,
            "pitch_std_hz": 4.23,
            "duration_sec": 0.66
          }
        },
        {
          "text": "동물매개팀의",
          "start": 3.56,
          "end": 4.78,
          "metrics": {
            "dB": -7.11,
            "pitch_mean_hz": 100.93,
            "pitch_std_hz": 13.14,
            "duration_sec": 1.22
          }
        },
        {
          "text": "조장",
          "start": 4.78,
          "end": 5.08,
          "metrics": {
            "dB": -6.03,
            "pitch_mean_hz": 90.39,
            "pitch_std_hz": 0.93,
            "duration_sec": 0.3
          }
        },
        {
          "text": "이태권입니다",
          "start": 5.08,
          "end": 5.78,
          "metrics": {
            "dB": -7.14,
            "pitch_mean_hz": 88.71,
            "pitch_std_hz": 2.96,
            "duration_sec": 0.7
          }
        },
        {
          "text": "반갑습니다",
          "start": 5.78,
          "end": 6.96,
          "metrics": {
            "dB": -13.43,
            "pitch_mean_hz": 76.43,
            "pitch_std_hz": 12.77,
            "duration_sec": 1.18
          }
        },
        {
          "text": "이번에",
          "start": 6.96,
          "end": 8.06,
          "metrics": {
            "dB": -22.34,
            "pitch_mean_hz": 73.21,
            "pitch_std_hz": 14.14,
            "duration_sec": 1.1
          }
        },
        {
          "text": "저희는",
          "start": 8.06,
          "end": 8.5,
          "metrics": {
            "dB": -2.9,
            "pitch_mean_hz": 99.16,
            "pitch_std_hz": 1.92,
            "duration_sec": 0.44
          }
        },
        {
          "text": "동물매개라는",
          "start": 8.5,
          "end": 9.54,
          "metrics": {
            "dB": -4.35,
            "pitch_mean_hz": 91.88,
            "pitch_std_hz": 5.33,
            "duration_sec": 1.04
          }
        },
        {
          "text": "주제를",
          "start": 9.54,
          "end": 10.34,
          "metrics": {
            "dB": -14.77,
            "pitch_mean_hz": 90.77,
            "pitch_std_hz": 1.18,
            "duration_sec": 0.8
          }
        },
        {
          "text": "가지고",
          "start": 10.34,
          "end": 10.74,
          "metrics": {
            "dB": -1.61,
            "pitch_mean_hz": 86.18,
            "pitch_std_hz": 2.37,
            "duration_sec": 0.4
          }
        },
        {
          "text": "활동을",
          "start": 10.74,
          "end": 11.24,
          "metrics": {
            "dB": -5.02,
            "pitch_mean_hz": 97.55,
            "pitch_std_hz": 4.57,
            "duration_sec": 0.5
          }
        },
        {
          "text": "했었는데요",
          "start": 11.24,
          "end": 11.8,
          "metrics": {
            "dB": -5.7,
            "pitch_mean_hz": 90.66,
            "pitch_std_hz": 11.5,
            "duration_sec": 0.56
          }
        }
      ]
    },
    {
      "sid": 1,
      "text": "활동을 하면서 간행을 겪기도 했었고 많은 어려움들이 있었습니다 하지만 우여곡절 끝에 이렇게 발표를 하게 되었고",
      "start": 12.6,
      "end": 21.3,
      "metrics": {
        "dB": -12.36,
        "pitch_mean_hz": 85.23,
        "pitch_std_hz": 12.82,
        "rate_wpm": 103.4,
        "pause_ratio": 0.006,
        "prosody_score": 12.75
      },
      "words": [
        {
          "text": "활동을",
          "start": 12.489999999999998,
          "end": 13.1,
          "metrics": {
            "dB": -10.85,
            "pitch_mean_hz": 88.5,
            "pitch_std_hz": 22.42,
            "duration_sec": 0.61
          }
        },
        {
          "text": "하면서",
          "start": 13.1,
          "end": 13.48,
          "metrics": {
            "dB": -1.86,
            "pitch_mean_hz": 88.03,
            "pitch_std_hz": 3.06,
            "duration_sec": 0.38
          }
        },
        {
          "text": "간행을",
          "start": 13.48,
          "end": 14.34,
          "metrics": {
            "dB": -12.56,
            "pitch_mean_hz": 79.76,
            "pitch_std_hz": 13.25,
            "duration_sec": 0.86
          }
        },
        {
          "text": "겪기도",
          "start": 14.34,
          "end": 14.78,
          "metrics": {
            "dB": -8.04,
            "pitch_mean_hz": 86.81,
            "pitch_std_hz": 0.71,
            "duration_sec": 0.44
          }
        },
        {
          "text": "했었고",
          "start": 14.78,
          "end": 15.32,
          "metrics": {
            "dB": -7.31,
            "pitch_mean_hz": 95.75,
            "pitch_std_hz": 10.81,
            "duration_sec": 0.54
          }
        },
        {
          "text": "많은",
          "start": 15.32,
          "end": 15.58,
          "metrics": {
            "dB": -2.41,
            "pitch_mean_hz": 88.58,
            "pitch_std_hz": 6.5,
            "duration_sec": 0.26
          }
        },
        {
          "text": "어려움들이",
          "start": 15.58,
          "end": 16.2,
          "metrics": {
            "dB": -3.73,
            "pitch_mean_hz": 91.08,
            "pitch_std_hz": 2.29,
            "duration_sec": 0.62
          }
        },
        {
          "text": "있었습니다",
          "start": 16.2,
          "end": 16.88,
          "metrics": {
            "dB": -4.59,
            "pitch_mean_hz": 82.32,
            "pitch_std_hz": 8.01,
            "duration_sec": 0.68
          }
        },
        {
          "text": "하지만",
          "start": 16.88,
          "end": 18.0,
          "metrics": {
            "dB": -21.72,
            "pitch_mean_hz": 74.21,
            "pitch_std_hz": 12.6,
            "duration_sec": 1.12
          }
        },
        {
          "text": "우여곡절",
          "start": 18.0,
          "end": 19.42,
          "metrics": {
            "dB": -11.4,
            "pitch_mean_hz": 82.52,
            "pitch_std_hz": 13.13,
            "duration_sec": 1.42
          }
        },
        {
          "text": "끝에",
          "start": 19.42,
          "end": 19.8,
          "metrics": {
            "dB": -6.08,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "duration_sec": 0.38
          }
        },
        {
          "text": "이렇게",
          "start": 19.8,
          "end": 20.08,
          "metrics": {
            "dB": -3.19,
            "pitch_mean_hz": 88.13,
            "pitch_std_hz": 5.14,
            "duration_sec": 0.28
          }
        },
        {
          "text": "발표를",
          "start": 20.08,
          "end": 20.52,
          "metrics": {
            "dB": -5.27,
            "pitch_mean_hz": 85.9,
            "pitch_std_hz": 6.01,
            "duration_sec": 0.44
          }
        },
        {
          "text": "하게",
          "start": 20.52,
          "end": 20.7,
          "metrics": {
            "dB": -0.8,
            "pitch_mean_hz": 93.94,
            "pitch_std_hz": 0.4,
            "duration_sec": 0.18
          }
        },
        {
          "text": "되었고",
          "start": 20.7,
          "end": 21.3,
          "metrics": {
            "dB": -7.84,
            "pitch_mean_hz": 92.72,
            "pitch_std_hz": 11.38,
            "duration_sec": 0.6
          }
        }
      ]
    },
    {
      "sid": 2,
      "text": "두 가지의 결과물을 가져왔습니다 그럼 지금부터 발표를 시작하겠습니다",
      "start": 22.0,
      "end": 26.98,
      "metrics": {
        "dB": -16.73,
        "pitch_mean_hz": 88.12,
        "pitch_std_hz": 14.38,
        "rate_wpm": 96.4,
        "pause_ratio": 0.007,
        "prosody_score": 14.27
      },
      "words": [
        {
          "text": "두",
          "start": 21.810000000000002,
          "end": 22.42,
          "metrics": {
            "dB": -12.42,
            "pitch_mean_hz": 66.17,
            "pitch_std_hz": 0.97,
            "duration_sec": 0.61
          }
        },
        {
          "text": "가지의",
          "start": 22.42,
          "end": 23.0,
          "metrics": {
            "dB": -5.2,
            "pitch_mean_hz": 100.71,
            "pitch_std_hz": 5.41,
            "duration_sec": 0.58
          }
        },
        {
          "text": "결과물을",
          "start": 23.0,
          "end": 23.6,
          "metrics": {
            "dB": -5.77,
            "pitch_mean_hz": 90.09,
            "pitch_std_hz": 6.6,
            "duration_sec": 0.6
          }
        },
        {
          "text": "가져왔습니다",
          "start": 23.6,
          "end": 24.32,
          "metrics": {
            "dB": -6.21,
            "pitch_mean_hz": 99.65,
            "pitch_std_hz": 21.27,
            "duration_sec": 0.72
          }
        },
        {
          "text": "그럼",
          "start": 24.32,
          "end": 25.3,
          "metrics": {
            "dB": -23.07,
            "pitch_mean_hz": 72.64,
            "pitch_std_hz": 12.95,
            "duration_sec": 0.98
          }
        },
        {
          "text": "지금부터",
          "start": 25.3,
          "end": 25.82,
          "metrics": {
            "dB": -4.38,
            "pitch_mean_hz": 93.52,
            "pitch_std_hz": 2.57,
            "duration_sec": 0.52
          }
        },
        {
          "text": "발표를",
          "start": 25.82,
          "end": 26.28,
          "metrics": {
            "dB": -2.79,
            "pitch_mean_hz": 90.56,
            "pitch_std_hz": 8.59,
            "duration_sec": 0.46
          }
        },
        {
          "text": "시작하겠습니다",
          "start": 26.28,
          "end": 26.98,
          "metrics": {
            "dB": -4.27,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "duration_sec": 0.7
          }
        }
      ]
    },
    {
      "sid": 3,
      "text": "감사합니다",
      "start": 26.98,
      "end": 27.14,
      "metrics": {
        "dB": -3.21,
        "pitch_mean_hz": 74.14,
        "pitch_std_hz": 1.63,
        "rate_wpm": 375.0,
        "pause_ratio": 0.01,
        "prosody_score": 1.61
      },
      "words": [
        {
          "text": "감사합니다",
          "start": 26.98,
          "end": 27.14,
          "metrics": {
            "dB": -3.21,
            "pitch_mean_hz": 74.14,
            "pitch_std_hz": 1.63,
            "duration_sec": 0.16
          }
        }
      ]
    }
  ]
}

```
