# -*- coding: utf-8 -*-
from typing import List, Dict, Any

# 한국어 문장 분할 라이브러리 (선택적)
try:
    import kss
    KSS_AVAILABLE = True
except ImportError:
    KSS_AVAILABLE = False
    print("⚠️ kss 라이브러리가 설치되지 않았습니다. 한국어 문장 분할이 비활성화됩니다.")


def map_llm_sections_to_sentences_with_timestamps(
    llm_sections: List[Dict[str, Any]],
    full_text: str,
    word_timestamps: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    LLM으로 분할된 문단을 kss로 문장 단위로 나누고, word timestamps를 사용하여 정확한 시간을 계산합니다.
    
    Args:
        llm_sections: LLM 분할 결과 (문단별)
        full_text: 전체 텍스트 (Whisper 결과)
        word_timestamps: Whisper의 word-level timestamps 리스트
    
    Returns:
        List[Dict]: 문장 단위로 분할된 segments (각 문장에 part, start, end, words 정보 포함)
    """
    if not llm_sections or not word_timestamps:
        return []
    
    # LLM 결과에서 sections 추출
    all_sections = []
    for item in llm_sections:
        if "sections" in item:
            all_sections.extend(item["sections"])
    
    if not all_sections:
        return []
    
    final_sentences = []
    
    for section in all_sections:
        section_text = section.get("content", "").strip()
        section_part = section.get("part", "")
        
        if not section_text:
            continue
        
        # kss로 문장 분할
        if KSS_AVAILABLE:
            try:
                sentences = kss.split_sentences(section_text)
            except Exception as e:
                print(f"⚠️ kss 문장 분할 실패: {e}, 문단 전체를 하나의 문장으로 처리")
                sentences = [section_text]
        else:
            # kss가 없으면 마침표 기준으로 분할
            sentences = _split_sentences_fallback(section_text)
        
        # 각 문장에 대해 word timestamps로 시간 계산
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 문장의 시간 정보 계산
            sentence_info = _calculate_sentence_timestamps(
                sentence,
                section_part,
                full_text,
                word_timestamps
            )
            
            if sentence_info:
                final_sentences.append(sentence_info)
    
    return final_sentences


def _split_sentences_fallback(text: str) -> List[str]:
    """
    kss가 없을 때 사용하는 fallback 문장 분할 함수.
    마침표, 느낌표, 물음표 기준으로 분할합니다.
    """
    import re
    if not text:
        return []
    
    # 문장 종결 기호로 분할
    sentences = re.split(r'([.!?]+)', text)
    
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
            if sentence:
                result.append(sentence)
    
    # 마지막 문장 처리
    if len(sentences) % 2 == 1:
        last_sentence = sentences[-1].strip()
        if last_sentence:
            result.append(last_sentence)
    
    return result if result else [text.strip()]


def _calculate_sentence_timestamps(
    sentence: str,
    part: str,
    full_text: str,
    word_timestamps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    문장 텍스트를 word timestamps와 매칭하여 정확한 시작/끝 시간을 계산합니다.
    
    Args:
        sentence: 문장 텍스트
        part: 문단 구분 (서론, 본론1, 본론2, 결론 등)
        full_text: 전체 텍스트
        word_timestamps: word-level timestamps 리스트
    
    Returns:
        Dict: {"text": 문장, "part": part, "start": 시작시간, "end": 끝시간, "words": [...]}
    """
    if not sentence or not word_timestamps:
        return None
    
    # 문장 텍스트 정규화 (공백 제거)
    sentence_normalized = sentence.replace(" ", "").replace("\n", "").replace("\t", "")
    
    # word timestamps에서 문장에 해당하는 words 찾기
    sentence_words = []
    current_text_normalized = ""
    word_start_idx = -1
    word_end_idx = -1
    
    for i, word_info in enumerate(word_timestamps):
        word_text = word_info.get("text", "").strip()
        word_text_normalized = word_text.replace(" ", "").replace("\n", "").replace("\t", "")
        
        # 현재까지 누적된 텍스트에 word 추가
        test_text = current_text_normalized + word_text_normalized
        
        # 문장의 시작 부분이 매칭되는지 확인
        if word_start_idx == -1:
            # 문장의 앞부분이 test_text에 포함되는지 확인
            if sentence_normalized.startswith(test_text) or test_text in sentence_normalized:
                word_start_idx = i
                current_text_normalized = test_text
            else:
                # 이전에 시작했지만 매칭 실패한 경우 리셋
                if word_start_idx != -1:
                    current_text_normalized = ""
                    word_start_idx = -1
        else:
            # 이미 시작했으면 계속 누적
            current_text_normalized = test_text
            
            # 문장 전체가 포함되었는지 확인
            if sentence_normalized in current_text_normalized or current_text_normalized in sentence_normalized:
                word_end_idx = i
                # 정확한 매칭을 위해 더 확인
                if len(current_text_normalized) >= len(sentence_normalized) * 0.8:  # 80% 이상 매칭
                    break
        
    # 매칭 실패 시 더 관대한 방법 시도
    if word_start_idx == -1 or word_end_idx == -1:
        # 문장의 일부 단어를 찾아서 시간 범위 추정
        sentence_words_list = sentence.split()
        if sentence_words_list:
            # 첫 단어와 마지막 단어 찾기
            first_word = sentence_words_list[0].strip()
            last_word = sentence_words_list[-1].strip()
            
            for i, word_info in enumerate(word_timestamps):
                word_text = word_info.get("text", "").strip()
                if first_word in word_text or word_text in first_word:
                    if word_start_idx == -1:
                        word_start_idx = i
                if last_word in word_text or word_text in last_word:
                    word_end_idx = i
                    if word_start_idx == -1:
                        word_start_idx = i
    
    # 최종 검증
    if word_start_idx == -1 or word_end_idx == -1 or word_start_idx > word_end_idx:
        print(f"⚠️ 문장 시간 계산 실패: '{sentence[:30]}...'")
        return None
    
    # 해당 범위의 words 추출
    sentence_words = []
    for i in range(word_start_idx, word_end_idx + 1):
        if i < len(word_timestamps):
            sentence_words.append(word_timestamps[i])
    
    if not sentence_words:
        return None
    
    # 시작/끝 시간 계산
    start_time = sentence_words[0].get("start", 0)
    end_time = sentence_words[-1].get("end", 0)
    
    return {
        "text": sentence,
        "part": part,
        "start": start_time,
        "end": end_time,
        "words": sentence_words
    }
