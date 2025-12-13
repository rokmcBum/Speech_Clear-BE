# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import requests
import json
import os
import uuid

load_dotenv()

class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        result_content = None
        collected_content = ""

        try:
            with requests.post(self._host + '/v1/chat-completions/HCX-003',
                               headers=headers, json=completion_request, stream=True, timeout=60) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if not line:
                        continue

                    decoded = line.decode("utf-8").strip()

                    # ✅ 스트리밍 종료
                    if decoded in ["data:[DONE]", "data: [DONE]"]:
                        break

                    # ✅ event:result인 경우에만 처리
                    if decoded.startswith("event:result"):
                        continue  # event 이름은 건너뜀

                    if decoded.startswith("data:"):
                        try:
                            data_json = json.loads(decoded.replace("data:", "").strip())

                            # ✅ event:result 데이터만 잡기
                            if data_json.get("message") and data_json["message"]["role"] == "assistant":
                                # 스트리밍 데이터 누적
                                content = data_json["message"].get("content", "")
                                if content:
                                    collected_content += content
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            print(f"⚠️ 데이터 파싱 에러: {e}")
                            pass

        except requests.exceptions.RequestException as e:
            print(f"⚠️ LLM API 요청 에러: {e}")
            return ""

        return collected_content if collected_content else ""

def get_sentence_feedback_from_LLM(sentence_info: dict):
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=os.getenv('LLM_API_Key'),
        request_id=str(uuid.uuid4())
    )

    preset_text = {
        "role":"system",
        "content":"""
        당신은 전문 스피치 코치입니다.

        아래 JSON은 한 문장에 대한 음성 분석 결과입니다.
        각 항목에는 (1) 라벨(label), (2) 해당 라벨에 대한 간단한 설명(comment)이 포함되어 있습니다.
        라벨 이름 자체를 해석하려고 하지 말고, comment에 적힌 내용을 바탕으로 자연스럽고 구체적인 피드백을 작성해주세요.

        [입력 JSON]
        {sentence_json}

        [작성 지침]

        1. 먼저 이 문장의 말하기 방식에서 전체적인 특징을 1문장 정도로 짚어주세요.
        2. 그 다음, volume_stability / volume_pattern / pitch_stability / pitch_ending / rate_level에 해당하는 comment들을 종합해서,
        이 문장에서 드러나는 말하기 습관과 아쉬운 점을 1~2문장으로 설명해주세요.
        3. comment가 없는 label의 경우에는 그 항목에 대해서는 언급하지 않아도 됩니다.
        4. 마지막으로, 연습할 때 바로 적용할 수 있는 구체적인 개선 방향을 1문장 정도로 제안해주세요.
        5. JSON 안의 label 이름이나 키 이름은 그대로 반복하지 말고, comment 내용을 자연스럽게 풀어서 말해주세요.
        6. 문체는 따뜻하고 격려하는 톤으로, 비난보다는 “이렇게 하면 더 좋아질 것 같다”는 방향으로 작성해주세요.
        7. 전체 길이는 3~4문장 정도로 작성하되, 각 문장을 별도 블록으로 나누어 출력해주세요.
        8. 답변은 반드시 한국어로 작성해주세요.
        9. 반복적인 표현과 내용을 피하고, 다양한 어휘를 사용해주세요.

        [출력 형식]

        각 문장은 아래와 같은 형태의 블록으로 출력합니다.

        <소제목1>
        첫 번째 문장 내용...

        <소제목2>
        두 번째 문장 내용...

        <소제목3>
        세 번째 문장 내용...

        (필요하다면 네 번째 문장도 같은 형식으로 추가)

        규칙:
        - 각 소제목은 3~8단어 정도의 짧은 한국어 문구로, 바로 아래 문장에서 말하고 싶은 핵심 키워드나 요지를 요약해 주세요.
        - 소제목에는 마침표를 쓰지 말고, 줄바꿈 후에 자연스러운 평서문으로 내용을 작성합니다.
        - 소제목과 본문 문장 사이에는 반드시 줄바꿈을 한 번 넣어 구분해 주세요.
        - 소제목에는 JSON의 라벨 이름을 직접 쓰지 말고, comment의 의미를 한 문구로 압축해 주세요.

        """
    }

    user_message = {
        "role": "user",
        "content": f"{sentence_info}"
    }

    request_data = {
        'messages': [preset_text,user_message],
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1500,
        'temperature': 0.5,
        'repeatPenalty': 1.05,
        'stopBefore': [],
        'includeAiFilters': True
    }

    # print(preset_text)
    # completion_executor.execute(request_data)
    answer = completion_executor.execute(request_data)
    # print(answer)

    # with open("LLM_result.json", "w", encoding="utf-8") as f:
    #     json.dump(answer, f, ensure_ascii=False, indent=2)

    return answer

def get_total_feedback_from_LLM(sentence_info: dict):
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=os.getenv('LLM_API_Key'),
        request_id=str(uuid.uuid4())
    )

    preset_text = {
        "role":"system",
        "content":"""
        당신은 전문 스피치 코치입니다.

        아래 JSON은 한 편의 발표(대본) 전체에 대한 음성 분석 결과를 요약한 것입니다.
        각 라벨은 이미 규칙 기반으로 평가된 것이므로, 수치나 기준을 다시 추론하지 말고
        라벨의 의미를 바탕으로 이 발표의 ‘전체적인 말하기 스타일’을 짧게 묘사해주세요.

        [입력 JSON]

        {script_json}

        각 필드의 의미는 다음과 같습니다.

        - volume_stability: 발표 전체에서 목소리 크기(음량)가 얼마나 안정적으로 유지되는지에 대한 평가입니다.
        - "stable": 전체적으로 볼륨 변화가 크지 않고, 비교적 일정하게 유지됨
        - "moderate": 적당한 범위 안에서 강약 변화가 있음
        - "unstable": 문장마다 볼륨 차이가 크고, 들쭉날쭉한 느낌을 줄 수 있음

        - volume_pattern: 발표 전반의 문장 끝 처리 패턴입니다.
        - "mostly_weak": 많은 문장에서 문장 끝으로 갈수록 힘이 빠지고, 마무리가 약해지는 경향
        - "mostly_clear": 대부분 문장에서 문장 끝이 또렷하게 유지됨
        - "mixed": 또렷한 마무리와 약한 마무리가 섞여 있음

        - pitch_stability: 발표 전체에서 목소리의 높낮이(피치)가 얼마나 다양하게 쓰였는지에 대한 평가입니다.
        - "flat": 높낮이 변화가 적어 단조롭게 들릴 수 있음
        - "balanced": 필요할 때 적절히 높낮이가 변하며, 과하지 않게 사용됨
        - "very_dynamic": 높낮이 변화가 큰 편으로, 활기 있지만 산만하게 느껴질 수도 있음

        - pitch_ending: 문장 끝에서 피치가 어떻게 변하는지에 대한 전반적인 경향입니다.
        - "mostly_falling": 대부분의 문장이 끝날 때 톤이 내려가며 마무리됨
        - "mostly_flat": 톤 변화 없이 평평하게 끝나는 문장이 많음
        - "many_rising": 톤이 올라가며 끝나는 문장이 자주 나타남
        - "mixed": 여러 패턴이 섞여 있음

        - rate_level: 발표 전체에서 전반적인 말하기 속도 수준입니다.
        - "mostly_slow": 전반적으로 느리게 말하는 편
        - "mostly_normal": 일반적인 속도 범위 안에서 말함
        - "mostly_fast": 전반적으로 빠른 편

        [작성 지침]

        1. 이 발표를 처음부터 끝까지 들었을 때 느껴질 전반적인 말투·분위기·리듬의 특징을 1~2문장으로만 요약해주세요.
        2. 개선 방향이나 조언은 쓰지 말고, 현재 말하기 스타일이 어떤 인상을 주는지 ‘묘사’하는 데에만 집중해주세요.
        3. 라벨 이름(예: "stable", "mostly_weak")이나 JSON 키 이름을 그대로 쓰지 말고,
        실제 청자가 들었을 때 느낄 만한 표현으로 자연스럽게 풀어서 설명해주세요.
        4. 답변은 반드시 한국어로 작성해주세요.
        5. 문장 수는 꼭 1~2문장을 넘기지 마세요.

        """
    }

    user_message = {
        "role": "user",
        "content": f"{sentence_info}"
    }

    request_data = {
        'messages': [preset_text,user_message],
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1500,
        'temperature': 0.5,
        'repeatPenalty': 1.05,
        'stopBefore': [],
        'includeAiFilters': True
    }

    # print(preset_text)
    # completion_executor.execute(request_data)
    answer = completion_executor.execute(request_data)
    # print(answer)

    # with open("LLM_result.json", "w", encoding="utf-8") as f:
    #     json.dump(answer, f, ensure_ascii=False, indent=2)

    return answer

def get_re_recording_feedback_from_LLM(original: dict, new: dict):
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=os.getenv('LLM_API_Key'),
        request_id=str(uuid.uuid4())
    )

    preset_text = {
        "role":"system",
        "content":"""
        당신은 전문 스피치 코치입니다.

        아래 JSON은 한 문장에 대해
        - 기존 녹음(original)
        - 재녹음(new)
        두 버전의 음성 분석 결과를 함께 담고 있습니다.

        각 항목에는 (1) 라벨(label), (2) 해당 라벨에 대한 간단한 설명(comment)이 포함되어 있습니다.
        라벨 이름 자체를 해석하려고 하지 말고, original과 new 각각의 comment 차이를 바탕으로
        변화 방향을 자연스럽고 구체적인 피드백으로 풀어주세요.

        [입력 JSON 예시]

        "original"
            "volume_stability": { "..." },
            "volume_pattern":   { "..." },
            "pitch_stability":  { "..." },
            "pitch_ending": { "..." },
            "rate_level": { "..." }
        
        "new"
            "volume_stability": { "..." },
            "volume_pattern":   { "..." },
            "pitch_stability":  { "..." },
            "pitch_ending": { "..." },
            "rate_level": { "..." }
        

        실제 입력에서는 위와 같은 구조로 {comparison_json}이 들어옵니다.

        [작성 지침]

        1. 먼저, 전체적으로 봤을 때 재녹음(new)이 기존(original)에 비해 어떻게 달라졌는지 한 문장 정도로 정리해주세요.
        - 좋아진 점(예: 말끝이 더 또렷해짐, 속도가 안정됨 등)을 우선적으로 짚어 주세요.
        - 크게 달라지지 않은 부분이 있다면 언급하지 마세요.

        2. 다음으로, volume_stability / volume_pattern / pitch_stability / pitch_ending / rate_level에 대한
        original과 new의 comment를 비교하여,
        - 어떤 점이 구체적으로 개선되었는지
        - 여전히 보완하면 좋을 아쉬운 지점이 있다면 무엇인지
        를 1~2문장으로 설명해주세요.

        3. 마지막으로, 재녹음을 기반으로 "다음 연습에서 집중하면 좋은 포인트"를 1문장 정도로 제안해주세요.
        - 예: "이제 말끝 힘이 좋아졌으니, 다음에는 속도 조절에 조금 더 신경 써보면 좋겠다"처럼
            변화 흐름을 이어가는 방향으로 작성해주세요.

        4. JSON 안의 label 이름(예: NORMAL_VAR, VOL_END_STABLE_CLEAR)이나 키 이름은 그대로 반복하지 말고,
        original/new의 comment 차이를 자연스럽게 요약해서 설명해주세요.

        5. 문체는 따뜻하고 격려하는 톤으로, 
        "이번에 이렇게 좋아졌다" → "다음에는 이런 부분을 더 살리면 좋겠다"라는 흐름을 유지해주세요.

        6. 전체 길이는 4~5문장 정도로 작성하되,
        각 문장을 별도 블록으로 나누어 아래 형식으로 출력해주세요.

        각 문장은 아래와 같은 형태의 블록으로 출력합니다.

        [출력 형식]
        <소제목1>
        첫 번째 문장 내용...

        <소제목2>
        두 번째 문장 내용...

        <소제목3>
        세 번째 문장 내용...

        (필요하다면 네 번째, 다섯 번째 문장도 같은 형식으로 추가)

        규칙:
        - 각 소제목은 3~8단어 정도의 짧은 한국어 문구로, 바로 아래 문장에서 말하고 싶은 핵심 변화나 요지를 요약해 주세요.
        (예: "말끝 처리의 개선", "여전히 남은 단조로움", "다음 연습의 핵심 포인트" 등)
        - 소제목에는 마침표를 쓰지 말고, 줄바꿈 후에 자연스러운 평서문으로 내용을 작성합니다.
        - 소제목과 본문 문장 사이에는 반드시 줄바꿈을 한 번 넣어 구분해 주세요.
        - 소제목에는 JSON의 라벨 이름을 직접 쓰지 말고, original과 new의 comment 차이를 한 문구로 압축해 주세요.

        [중요]
        - 이번 피드백의 초점은 "new가 original에서 어떻게 변화했는가"입니다.
        - 단순히 두 버전을 각각 평가하는 것이 아니라,
        변화 방향(개선 / 유지 / 아쉬운 부분)을 중심으로 설명해 주세요.
        """
    }

    user_message = {
        "role": "user",
        "content": f"original\n{original}\n\nnew\n{new}"
    }

    request_data = {
        'messages': [preset_text,user_message],
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1500,
        'temperature': 0.5,
        'repeatPenalty': 1.05,
        'stopBefore': [],
        'includeAiFilters': True
    }

    # print(preset_text)
    # completion_executor.execute(request_data)
    print(f"[DEBUG] get_re_recording_feedback_from_LLM 호출됨")
    print(f"[DEBUG] original: {original}")
    print(f"[DEBUG] new: {new}")
    
    answer = completion_executor.execute(request_data)
    
    print(f"[DEBUG] LLM execute 결과: type={type(answer)}, value={answer[:200] if answer else 'None'}")

    # with open("LLM_result.json", "w", encoding="utf-8") as f:
    #     json.dump(answer, f, ensure_ascii=False, indent=2)

    # answer가 None이거나 빈 문자열인 경우 처리
    if answer is None:
        print(f"[WARN] get_re_recording_feedback_from_LLM: LLM 응답이 None입니다.")
        return ""
    
    if isinstance(answer, str) and answer.strip() == "":
        print(f"[WARN] get_re_recording_feedback_from_LLM: LLM 응답이 빈 문자열입니다.")
        return ""

    return answer