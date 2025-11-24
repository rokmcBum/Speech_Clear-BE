# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Any


def map_llm_sections_to_whisper_segments(
    llm_sections: List[Dict[str, Any]],
    whisper_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    LLM으로 분할된 문단을 Whisper segments와 매핑하여 시간 정보를 추가합니다.
    간단한 로직: 문단 텍스트가 포함된 첫 번째와 마지막 whisper segment를 찾습니다.
    
    Args:
        llm_sections: LLM 분할 결과
        whisper_segments: Whisper 분석 결과 segments (문장별로 분리됨)
    
    Returns:
        List[Dict]: 시간 정보가 매핑된 segments (문단별로 분리됨)
    """
    if not llm_sections or not whisper_segments:
        return []
    
    # LLM 결과에서 sections 추출
    all_sections = []
    for item in llm_sections:
        if "sections" in item:
            all_sections.extend(item["sections"])
    
    if not all_sections:
        return []
    
    # Whisper segments의 텍스트를 순서대로 합쳐서 전체 대본 재구성
    whisper_texts = [seg["text"].strip() for seg in whisper_segments]
    whisper_full_text = " ".join(whisper_texts)
    whisper_full_text_normalized = whisper_full_text.replace(" ", "").replace("\n", "").replace("\t", "")
    
    mapped_segments = []
    whisper_idx = 0  # 현재 처리 중인 Whisper segment 인덱스
    
    for section in all_sections:
        section_text = section.get("content", "").strip()
        if not section_text:
            continue
        
        # 문단 텍스트를 정규화 (공백 제거)
        section_text_normalized = section_text.replace(" ", "").replace("\n", "").replace("\t", "")
        
        if not section_text_normalized:
            continue
        
        # 문단 텍스트가 전체 텍스트에 포함되는지 확인
        if section_text_normalized not in whisper_full_text_normalized:
            print(f"⚠️ 문단 ({section.get('part', '')}) 텍스트가 Whisper 텍스트에 없음, 스킵")
            continue
        
        # 문단의 시작 위치 찾기 (whisper_idx부터)
        start_idx = None
        accumulated_normalized = ""
        
        for i in range(whisper_idx, len(whisper_segments)):
            seg_text = whisper_segments[i]["text"].strip()
            accumulated_normalized += seg_text.replace(" ", "").replace("\n", "").replace("\t", "")
            
            # 문단의 앞부분이 포함되는지 확인
            if len(accumulated_normalized) >= len(section_text_normalized):
                if section_text_normalized in accumulated_normalized:
                    # 문단이 시작되는 첫 segment 찾기
                    for j in range(whisper_idx, i + 1):
                        test_accumulated = "".join([
                            seg["text"].strip().replace(" ", "").replace("\n", "").replace("\t", "")
                            for seg in whisper_segments[j:i+1]
                        ])
                        if section_text_normalized in test_accumulated:
                            start_idx = j
                            break
                    if start_idx is not None:
                        break
        
        # 전체 텍스트에서 직접 찾기 (위 방법 실패 시)
        if start_idx is None:
            char_pos = whisper_full_text_normalized.find(section_text_normalized)
            if char_pos >= 0:
                # char_pos를 segment 인덱스로 변환
                char_count = 0
                for i, seg in enumerate(whisper_segments):
                    seg_len = len(seg["text"].strip().replace(" ", "").replace("\n", "").replace("\t", ""))
                    if char_count + seg_len > char_pos:
                        start_idx = max(whisper_idx, i)
                        break
                    char_count += seg_len
        
        if start_idx is None:
            print(f"⚠️ 문단 ({section.get('part', '')}) 시작 위치를 찾을 수 없음, whisper_idx 사용")
            start_idx = whisper_idx
        
        # 범위 체크
        if start_idx >= len(whisper_segments):
            start_idx = len(whisper_segments) - 1
        
        # 문단의 끝 위치 찾기
        end_idx = start_idx
        accumulated_normalized = ""
        
        for i in range(start_idx, len(whisper_segments)):
            seg_text = whisper_segments[i]["text"].strip()
            accumulated_normalized += seg_text.replace(" ", "").replace("\n", "").replace("\t", "")
            
            # 문단 전체가 포함되었는지 확인
            if section_text_normalized in accumulated_normalized:
                end_idx = i
                break
        
        # 범위 체크
        if end_idx >= len(whisper_segments):
            end_idx = len(whisper_segments) - 1
        
        # 유효성 검사
        if start_idx < 0 or end_idx < 0 or start_idx >= len(whisper_segments) or end_idx >= len(whisper_segments):
            print(f"⚠️ 문단 ({section.get('part', '')}) 인덱스 범위 오류, 스킵")
            continue
        
        # 매핑된 segments의 시간 정보 사용
        start_seg = whisper_segments[start_idx]
        end_seg = whisper_segments[end_idx]
        
        # 해당 구간의 모든 words 수집
        all_words = []
        for i in range(start_idx, end_idx + 1):
            all_words.extend(whisper_segments[i].get("words", []))
        
        # 시간 정보만 매핑 (metrics는 나중에 librosa로 계산)
        mapped_segments.append({
            "text": section_text,
            "part": section.get("part", ""),
            "start": start_seg["start"],
            "end": end_seg["end"],
            "words": all_words
        })
        
        # 다음 문단을 위해 whisper_idx 업데이트
        whisper_idx = min(end_idx + 1, len(whisper_segments))
    
    return mapped_segments


def split_paragraph_into_sentences(paragraph_text: str) -> List[str]:
    """
    문단을 문장 단위로 분할합니다.
    한국어 기준: 마침표(.), 느낌표(!), 물음표(?)로 분할
    
    Args:
        paragraph_text: 분할할 문단 텍스트
    
    Returns:
        List[str]: 문장 리스트
    """
    if not paragraph_text:
        return []
    
    # 문장 종결 기호로 분할 (., !, ?)
    # 정규표현식으로 문장 종결 기호와 함께 분할
    sentences = re.split(r'([.!?]+)', paragraph_text)
    
    # 분할된 결과를 문장과 종결 기호를 합쳐서 재구성
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
            if sentence:
                result.append(sentence)
        else:
            # 마지막 문장 (종결 기호 없을 수 있음)
            sentence = sentences[i].strip()
            if sentence:
                result.append(sentence)
    
    # 마지막 문장 처리 (홀수 개인 경우)
    if len(sentences) % 2 == 1:
        last_sentence = sentences[-1].strip()
        if last_sentence:
            result.append(last_sentence)
    
    return result if result else [paragraph_text.strip()]


def map_sentences_to_whisper_segments(
    paragraph_text: str,
    paragraph_part: str,
    paragraph_start_time: float,
    paragraph_end_time: float,
    whisper_segments: List[Dict[str, Any]],
    used_segment_indices: set = None
) -> List[Dict[str, Any]]:
    """
    문단 범위 내의 Whisper segments를 문장 단위 segment로 변환합니다.
    Whisper segments는 이미 문장 단위로 나뉘어져 있으므로, 문단 범위 내의 segments를 그대로 사용합니다.
    각 segment는 한 번만 사용되도록 보장합니다.
    
    Args:
        paragraph_text: 문단 텍스트 (참고용, 실제로는 사용하지 않음)
        paragraph_part: 문단 구분 (서론, 본론1, 본론2, 결론 등)
        paragraph_start_time: 문단 시작 시간
        paragraph_end_time: 문단 끝 시간
        whisper_segments: Whisper 분석 결과 segments (문장별로 분리됨)
        used_segment_indices: 이미 사용된 segment 인덱스 집합 (중복 방지용)
    
    Returns:
        List[Dict]: 문장 단위로 분할된 segments (각 문장에 part 정보와 시간 정보 포함)
    """
    if not whisper_segments:
        return []
    
    if used_segment_indices is None:
        used_segment_indices = set()
    
    # 문단 시간 범위 내의 Whisper segments 찾기
    sentence_segments = []
    
    for idx, seg in enumerate(whisper_segments):
        # 이미 사용된 segment는 건너뛰기
        if idx in used_segment_indices:
            continue
        
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        
        # 문단 시간 범위와 겹치는 segment인지 확인
        # segment가 문단 범위 내에 있거나, 일부라도 겹치면 포함
        if (seg_start >= paragraph_start_time and seg_start <= paragraph_end_time) or \
           (seg_end >= paragraph_start_time and seg_end <= paragraph_end_time) or \
           (seg_start <= paragraph_start_time and seg_end >= paragraph_end_time):
            
            # words 수집
            words = seg.get("words", [])
            
            sentence_segments.append({
                "text": seg.get("text", "").strip(),
                "part": paragraph_part,  # 문단 정보 추가
                "start": seg_start,
                "end": seg_end,
                "words": words
            })
            
            # 사용된 segment 인덱스에 추가
            used_segment_indices.add(idx)
    
    return sentence_segments

