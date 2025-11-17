# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import requests
import json
import os
import re
from typing import List, Dict, Any

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
        
        with requests.post(self._host + '/v3/chat-completions/HCX-005',
                           headers=headers, json=completion_request, stream=True) as r:
            r.raise_for_status()  # HTTP 에러 체크
            
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
                            message = data_json["message"]
                            
                            # 델타 방식 처리 (delta.content가 있으면 델타, 없으면 전체)
                            if "delta" in message and "content" in message["delta"]:
                                # 델타 방식: 조각조각 누적
                                delta_content = message["delta"].get("content", "")
                                if delta_content:
                                    if result_content is None:
                                        result_content = delta_content
                                    else:
                                        result_content += delta_content
                            else:
                                # 전체 응답 방식: content가 있으면 사용
                                content = message.get("content", "")
                                if content:
                                    # 중복 방지: 이미 같은 내용이 있으면 스킵
                                    if result_content and content in result_content:
                                        continue
                                    # 새로운 완전한 응답이면 덮어쓰기
                                    result_content = content
                    except json.JSONDecodeError:
                        pass

        return result_content


def classify_text_into_sections(text: str) -> List[Dict[str, Any]]:
    """
    텍스트를 LLM을 사용하여 문단별(서론/본론/결론)로 분할합니다.
    
    Args:
        text: 분할할 텍스트 (전체 대본)
    
    Returns:
        List[Dict]: 분할된 섹션 리스트
        [
            {
                "title": "대본 주제",
                "sections": [
                    {
                        "id": 1,
                        "part": "서론",
                        "content": "대본 원문 일부"
                    },
                    ...
                ]
            }
        ]
    """
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=os.getenv('LLM_API_Key'),
        request_id='7cb606aa44574fc0b35cc2c620f5e7d3'
    )

    preset_text = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """
당신은 전문적인 한국어 스피치 코치입니다. 발표나 면접 등 다양한 상황에서 화자의 대본을 **편집하지 않고 그대로 보존한 채**, 내용별로 문단을 나누는 것이 당신의 역할입니다.

⚠️ 반드시 지켜야 할 원칙:
1. 사용자가 제공한 대본의 **단어, 문장, 띄어쓰기, 구두점, 줄바꿈**을 절대로 바꾸지 마세요.
2. **어떤 문장도 삭제, 요약, 수정하지 마세요.**
3. 대본의 모든 텍스트를 **순서대로 100% 포함해야 합니다.**
4. 오직 "section" 구분만 수행합니다. (서론, 본론, 결론)
5. "content" 항목에는 대본의 **원문 그대로** 포함합니다.
6. JSON 외의 문장, 설명, 코드블록(````json 등)은 절대 출력하지 마세요.

---

[출력 형식 예시]
[
  {
    "title": "대본 주제",
    "sections": [
        {
        "id": 1,
        "part": "서론",
        "content": "대본 원문 일부 (수정 없이)"
        },
        {
            "id": 2,
            "part": "본론1",
            "content": "대본 원문 일부 (수정 없이)"
        },
        {
            "id": 3,
            "part": "본론2",
            "content": "대본 원문 일부 (수정 없이)"
        },
        ...

        {
        "id": 4,
        "part": "결론",
        "content": "대본 원문 일부 (수정 없이)"
        }
    ]
  }
]

---

이제 곧 사용자가 줄 대본을 그대로 분석하여, **수정하지 않고 구간만 나눈 JSON**을 출력하세요.
대본의 단어 하나라도 바꾸면 실패로 간주됩니다. 반드시 원문 그대로 출력하세요.
"""
            }
        ]
    }

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text
            }
        ]
    }
    
    request_data = {
        'messages': [preset_text, user_message],
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1024,
        'temperature': 0.5,
        'repeatPenalty': 1.1,
        'stopBefore': [],
        'includeAiFilters': True
    }

    answer = completion_executor.execute(request_data)
    
    if not answer:
        raise ValueError("LLM 응답이 비어있습니다.")
    
    # JSON 파싱 전에 코드블록 제거 및 포맷팅
    cleaned = answer.strip()
    
    # 중복 제거: 같은 JSON이 여러 번 반복되는 경우 첫 번째만 사용
    # ```json ... ``` 패턴이 여러 번 있으면 첫 번째만 추출
    code_blocks = list(re.finditer(r'`{3,}(?:json)?\s*\n', cleaned, re.IGNORECASE))
    if len(code_blocks) > 1:
        # 첫 번째 코드블록만 사용
        first_block = code_blocks[0]
        second_block = code_blocks[1]
        # 첫 번째 코드블록 시작부터 두 번째 코드블록 시작 전까지
        start_idx = first_block.end()
        cleaned = cleaned[start_idx:second_block.start()].strip()
        # 마지막 ``` 제거 (혹시 남아있을 수 있음)
        cleaned = re.sub(r'`{3,}\s*$', '', cleaned, flags=re.MULTILINE).strip()
    elif "```" in cleaned:
        # 단일 코드블록 제거
        start_match = re.search(r'`{3,}(?:json)?', cleaned, re.IGNORECASE)
        if start_match:
            start_idx = start_match.end()
            # json 키워드가 있으면 다음 줄부터
            if "json" in cleaned[start_match.start():start_idx].lower():
                newline_idx = cleaned.find("\n", start_idx)
                if newline_idx != -1:
                    start_idx = newline_idx + 1
            
            # 마지막 코드블록 끝 찾기
            end_matches = list(re.finditer(r'`{3,}', cleaned))
            if end_matches:
                last_match = end_matches[-1]
                if last_match.start() > start_idx:
                    end_idx = last_match.start()
                    cleaned = cleaned[start_idx:end_idx].strip()
    
    # JSON 배열/객체만 추출 ([ 또는 { 로 시작하는 부분)
    json_start = -1
    for char in ['[', '{']:
        idx = cleaned.find(char)
        if idx != -1 and (json_start == -1 or idx < json_start):
            json_start = idx
    
    if json_start != -1:
        # 중괄호/대괄호 매칭하여 끝 찾기
        stack = []
        json_end = -1
        for i in range(json_start, len(cleaned)):
            char = cleaned[i]
            if char in ['[', '{']:
                stack.append(char)
            elif char in [']', '}']:
                if stack:
                    stack.pop()
                    if not stack:
                        json_end = i + 1
                        break
        
        if json_end != -1:
            cleaned = cleaned[json_start:json_end].strip()
    
    # 3. 최종 파싱
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 파싱 실패")
        print(f"원본 응답 (처음 500자): {answer[:500]}")
        print(f"정리된 응답 (처음 500자): {cleaned[:500]}")
        raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {e}")