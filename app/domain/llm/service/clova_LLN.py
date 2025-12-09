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

        result_content = None  # 초기화

        with requests.post(self._host + '/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
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
                            # 최종 결과 저장
                            result_content = data_json["message"].get("content", None)
                    except json.JSONDecodeError:
                        pass

        return result_content if result_content is not None else ""

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
        7. 전체 길이는 3~4문장 정도의 하나의 단락으로 작성해주세요.
        8. 답변은 반드시 한국어로 작성해주세요.
        9. 반복적인 표현과 내용을 피하고, 다양한 어휘를 사용해주세요.

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

def get_paragraph_feedback_from_LLM(sentence_info: dict):
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=os.getenv('LLM_API_Key'),
        request_id=str(uuid.uuid4())
    )

    preset_text = {
        "role":"system",
        "content":"""
        당신은 전문 스피치 코치입니다.

        아래 JSON은 하나의 문단에 대한 음성 분석 결과를 사람이 해석하기 쉽게 라벨로 정리한 것입니다.
        각 라벨은 이미 규칙 기반으로 평가된 것이므로, 수치나 기준을 다시 추론하려고 하지 말고,
        라벨의 의미를 바탕으로 자연스럽고 구체적인 피드백을 작성해주세요.

        [입력 JSON]

        {paragraph_json}

        각 필드의 의미는 다음과 같습니다.

        - part: 이 문단이 스크립트에서 어떤 역할을 하는지 나타냅니다.
        - "서론": 발표의 시작 부분, 주제 소개와 전체 흐름 안내를 담당하는 문단
        - "본론": 핵심 내용과 설명이 들어가는 문단
        - "결론": 내용을 정리하고 메시지를 다시 강조하는 마무리 문단

        - volume_stability: 이 문단에서 목소리 크기(음량)가 얼마나 안정적으로 유지되는지에 대한 평가입니다.
        - "stable": 전체적으로 볼륨 변화가 크지 않고, 비교적 일정하게 유지됨
        - "moderate": 적당한 범위 안에서 강약 변화가 있음
        - "unstable": 문장마다 볼륨 차이가 크고, 들쭉날쭉한 느낌을 줄 수 있음

        - volume_pattern: 이 문단의 문장 끝 처리 패턴입니다.
        - "mostly_weak": 많은 문장에서 문장 끝으로 갈수록 힘이 빠지고, 마무리가 약해지는 경향
        - "mostly_clear": 대부분 문장에서 문장 끝이 또렷하게 유지됨
        - "mixed": 또렷한 마무리와 약한 마무리가 섞여 있음

        - pitch_stability: 이 문단에서 목소리의 높낮이(피치)가 얼마나 다양하게 쓰였는지에 대한 평가입니다.
        - "flat": 높낮이 변화가 적어 단조롭게 들릴 수 있음
        - "balanced": 필요할 때 적절히 높낮이가 변하며, 과하지 않게 사용됨
        - "very_dynamic": 높낮이 변화가 큰 편으로, 활기 있지만 산만하게 느껴질 수도 있음

        - pitch_ending: 문장 끝에서 피치가 어떻게 변하는지에 대한 전반적인 경향입니다.
        - "mostly_falling": 대부분의 문장이 끝날 때 톤이 내려가며 마무리됨
        - "mostly_flat": 톤 변화 없이 평평하게 끝나는 문장이 많음
        - "many_rising": 톤이 올라가며 끝나는 문장이 자주 나타남
        - "mixed": 여러 패턴이 섞여 있음

        - rate_level: 이 문단에서 전반적인 말하기 속도 수준입니다.
        - "mostly_slow": 전반적으로 느리게 말하는 편
        - "mostly_normal": 일반적인 속도 범위 안에서 말함
        - "mostly_fast": 전반적으로 빠른 편

        [역할별(서론/본론/결론) 피드백 관점]

        - part이 "서론"인 경우:
        - 청자가 주제와 흐름을 이해할 수 있도록, 시작 부분의 안정감과 속도가 적절한지 언급해주세요.
        - 너무 빠르게 시작하거나, 처음부터 힘이 너무 세거나 약하지 않은지 평가해주세요.
        - 발표 전체에 대한 기대감을 만들어 주는지, 도입부로서 분위기가 잘 잡혀 있는지도 함께 언급해주세요.

        - part이 "본론"인 경우:
        - 핵심 내용을 설명하는 부분이므로, 정보 전달에 적합한 속도와 강약 조절이 이루어지는지 평가해주세요.
        - 중요한 내용이 충분히 강조되고 있는지, 문장 끝 마무리와 억양이 내용을 잘 살려주는지 언급해주세요.
        - 설명이 길어질 때도 에너지가 유지되는지, 중간에 힘이 빠지지 않는지에 대한 코멘트도 포함해주세요.

        - part이 "결론"인 경우:
        - 결론은 발표의 인상을 결정하는 부분입니다.
        - 마지막 문장들의 안정감·명확함·자신감 여부를 특히 강조해주세요.
        - 문단 후반에서 힘이 빠지는지, 메시지가 흐려지는지 등을 자연스럽게 설명해주세요.
        - 개선 방향은 가능하면 긍정적이고 밝은 톤으로 제안해주세요.

        [작성 지침]

        1. 먼저 이 문단의 장점을 1~2문장 정도로 짚어주세요.  
        2. 그 다음, volume_stability / volume_pattern / pitch_stability / pitch_ending / rate_level 라벨을 바탕으로
        이 문단에서 보이는 말하기 습관과 한계를 구체적으로 설명해주세요.
        3. 실제 발표자가 연습할 때 참고할 수 있는 개선 방향을 1~2가지 제안해주세요.
        4. 라벨 이름(예: "stable", "mostly_weak")이나 JSON 키 이름을 그대로 쓰지 말고,
        청자가 들을 때 어떻게 느낄지를 중심으로 풀어서 설명해주세요.
        5. 문체는 따뜻하고 격려하는 말투로 작성하고, 비난보다는 “이렇게 하면 더 좋아질 것 같다”는 방향으로 적어주세요.
        6. 전체 길이는 4~6문장 정도의 하나의 문단으로 작성해주세요.
        7. 답변은 반드시 한국어로 작성해주세요.

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