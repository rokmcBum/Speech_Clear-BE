# -*- coding: utf-8 -*-
from typing import List, Dict, Any

# 한국어 문장 분할 라이브러리 (선택적)
try:
    import kss
    KSS_AVAILABLE = True
except ImportError:
    KSS_AVAILABLE = False
    print("⚠️ kss 라이브러리가 설치되지 않았습니다. 한국어 문장 분할이 비활성화됩니다.")


def split_llm_sections_into_sentences_with_clova_timestamps(
    llm_sections: List[Dict[str, Any]],
    clova_words: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not llm_sections or not clova_words:
        return []
    
    # LLM 결과에서 sections 추출
    all_sections = []
    for item in llm_sections:
        if "sections" in item:
            all_sections.extend(item["sections"])
    
    if not all_sections:
        return []
    
    final_sentences = []
    last_end_time = 0.0  # 이전 문장의 끝 시간 (초 단위)
    
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
        
        # 각 문장에 대해 Clova word timestamps로 시간 계산 (순차적으로 처리)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 문장의 시간 정보 계산 (이전 문장의 끝 시간 이후부터 검색)
            sentence_info = _calculate_sentence_timestamps_from_clova_words(
                sentence,
                section_part,
                clova_words,
                min_start_time=last_end_time  # 이전 문장의 끝 시간 이후부터 검색
            )
            
            if sentence_info:
                # 밀리초를 초로 변환하여 비교
                seg_start = sentence_info["start"] / 1000.0
                seg_end = sentence_info["end"] / 1000.0
                
                # 겹침 방지: 이전 문장의 끝 시간보다 이전에 시작하면 조정
                if seg_start < last_end_time:
                    # 이전 문장의 끝 시간 이후로 시작 시간 조정
                    seg_start = last_end_time
                    sentence_info["start"] = int(seg_start * 1000)
                
                # 끝 시간도 시작 시간보다 이후여야 함
                if seg_end <= seg_start:
                    # 최소 0.1초 길이 보장
                    seg_end = seg_start + 0.1
                    sentence_info["end"] = int(seg_end * 1000)
                
                last_end_time = seg_end
                final_sentences.append(sentence_info)
    
    return final_sentences


def _calculate_sentence_timestamps_from_clova_words(
    sentence: str,
    part: str,
    clova_words: List[Dict[str, Any]],
    min_start_time: float = 0.0
) -> Dict[str, Any]:
    """
    문장 텍스트를 Clova Speech의 word timestamps와 매칭하여 정확한 시작/끝 시간을 계산합니다.
    
    Args:
        sentence: 문장 텍스트
        part: 문단 구분 (서론, 본론1, 본론2, 결론 등)
        clova_words: Clova Speech의 word timestamps 리스트 [{"text": "...", "start": 0.0, "end": 1.0}]
    
    Returns:
        Dict: {"text": 문장, "part": part, "start": 시작시간(초), "end": 끝시간(초), "words": [...]}
    """
    if not sentence or not clova_words:
        return None
    
    # 문장의 단어 리스트
    sentence_words_list = sentence.split()
    if not sentence_words_list:
        return None
    
    # Clova words에서 문장에 해당하는 words 찾기
    # 문장의 모든 단어를 순차적으로 매칭하는 방식 사용
    word_start_idx = -1
    word_end_idx = -1
    
    # 문장의 단어들을 정규화 (공백, 구두점 제거)
    sentence_words_normalized = [
        word.strip().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        for word in sentence_words_list
    ]
    
    # min_start_time 이후부터 시작하는 인덱스 찾기
    start_search_idx = 0
    for i, word_info in enumerate(clova_words):
        if word_info.get("start", 0.0) >= min_start_time:
            start_search_idx = i
            break
    
    # 문장의 단어들을 순차적으로 매칭
    sent_word_idx = 0  # 문장 내 현재 단어 인덱스
    consecutive_matches = 0  # 연속 매칭 횟수
    best_match_start = -1
    best_match_end = -1
    best_consecutive = 0
    
    for i in range(start_search_idx, len(clova_words)):
        if sent_word_idx >= len(sentence_words_normalized):
            break
        
        clova_word = clova_words[i]
        clova_text = clova_word.get("text", "").strip()
        clova_text_normalized = clova_text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        
        target_word = sentence_words_normalized[sent_word_idx]
        
        # 단어 매칭 (부분 문자열 또는 완전 일치)
        if (target_word in clova_text_normalized or 
            clova_text_normalized in target_word or
            target_word == clova_text_normalized):
            
            if word_start_idx == -1:
                word_start_idx = i
            word_end_idx = i
            consecutive_matches += 1
            sent_word_idx += 1
            
            # 최고 연속 매칭 업데이트
            if consecutive_matches > best_consecutive:
                best_consecutive = consecutive_matches
                best_match_start = word_start_idx
                best_match_end = word_end_idx
        else:
            # 매칭 실패 시 연속 매칭 리셋
            if consecutive_matches > 0:
                # 이전까지의 매칭이 최고 기록이면 저장
                if consecutive_matches > best_consecutive:
                    best_consecutive = consecutive_matches
                    best_match_start = word_start_idx
                    best_match_end = word_end_idx
                
                # 현재 단어부터 다시 시작
                consecutive_matches = 0
                word_start_idx = -1
                word_end_idx = -1
                sent_word_idx = 0
    
    # 최고 매칭 결과 사용
    if best_match_start != -1 and best_match_end != -1:
        word_start_idx = best_match_start
        word_end_idx = best_match_end
    
    # 최종 검증
    if word_start_idx == -1 or word_end_idx == -1 or word_start_idx > word_end_idx:
        print(f"⚠️ 문장 시간 계산 실패: '{sentence[:30]}...'")
        return None
    
    # 해당 범위의 words 추출 (Clova 형식으로 변환: [start_ms, end_ms, text])
    sentence_clova_words = []
    for i in range(word_start_idx, word_end_idx + 1):
        if i < len(clova_words):
            word_info = clova_words[i]
            # Clova 형식으로 변환 (밀리초 단위)
            sentence_clova_words.append([
                int(word_info.get("start", 0) * 1000),  # 초 -> 밀리초
                int(word_info.get("end", 0) * 1000),    # 초 -> 밀리초
                word_info.get("text", "").strip()
            ])
    
    if not sentence_clova_words:
        return None
    
    # 시작/끝 시간 계산 (밀리초 단위)
    start_time_ms = int(clova_words[word_start_idx].get("start", 0) * 1000)
    end_time_ms = int(clova_words[word_end_idx].get("end", 0) * 1000)
    
    return {
        "text": sentence,
        "part": part,
        "start": start_time_ms,  # 밀리초
        "end": end_time_ms,      # 밀리초
        "words": sentence_clova_words  # [start_ms, end_ms, text] 형태
    }


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