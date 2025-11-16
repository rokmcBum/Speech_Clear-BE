# -*- coding: utf-8 -*-
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

