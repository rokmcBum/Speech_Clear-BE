from dotenv import load_dotenv
import requests
import json
import os

load_dotenv()

class ClovaSpeechClient:
    # Clova Speech invoke URL
    invoke_url = os.getenv('Clova_Speech_Invoke_URL')
    # Clova Speech secret key
    secret = os.getenv('Clova_Speech_Secret_Key')

    def req_url(self, url, completion, callback=None, userdata=None,
    	forbiddens=None, boostings=None, wordAlignment=True,
        	fullText=True, diarization=None, sed=None):
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_object_storage(self, data_key, completion, callback=None,
    	userdata=None, forbiddens=None, boostings=None,wordAlignment=True,
        	fullText=True, diarization=None, sed=None):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_upload(self, file, completion, callback=None, userdata=None,
    	forbiddens=None, boostings=None, wordAlignment=True, 
        	fullText=True, diarization=None, sed=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        print(json.dumps(request_body, ensure_ascii=False).encode('UTF-8'))
        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body,
            			ensure_ascii=False).encode('UTF-8'),
                        		'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url
        			+ '/recognizer/upload', files=files)
        return response

def make_voice_to_stt(audio_file_path: str):
    import librosa
    
    res = ClovaSpeechClient().req_upload(file=audio_file_path, completion='sync')
    
    if res.status_code != 200:
        print(f"❌ Clova Speech API 요청 실패: 상태 코드 {res.status_code}")
        print(f"응답 내용:\n{res.text}")
        raise Exception(f"Clova Speech API 요청 실패: {res.status_code}")
    
    try:
        result = res.json()
    except json.JSONDecodeError:
        print(f"❌ JSON 디코딩 오류: 응답 텍스트를 JSON으로 변환할 수 없습니다.")
        print(f"응답 텍스트:\n{res.text}")
        raise Exception("Clova Speech API 응답 파싱 실패")
    
    # 디버깅: 응답 구조 확인
    print(f"🔍 Clova Speech API 응답 구조:")
    print(f"   - result keys: {list(result.keys())}")
    print(f"   - result type: {type(result)}")
    print(f"   - 전체 응답 (일부): {json.dumps(result, ensure_ascii=False, indent=2)[:500]}")
    
    # 전체 텍스트 추출 (여러 가능한 키 확인)
    full_text = result.get("text", "") or result.get("fullText", "") or result.get("transcript", "")
    
    # segments에서 words 추출
    segments_data = result.get("segments", [])
    
    print(f"   - full_text: '{full_text[:50]}...' (길이: {len(full_text)})")
    print(f"   - segments 개수: {len(segments_data)}")
    all_words = []
    
    # 2. segments 배열을 순회하며 'diarization' 및 'speaker' 필드 제거
    final_segments = []
    for seg in segments_data:
        # 딕셔너리 복사 (원본 데이터 보호)
        clean_seg = seg.copy() 
        
        # 'speaker' 정보가 포함된 필드 삭제
        if 'speaker' in clean_seg:
            del clean_seg['speaker']
        
        # 'diarization' 정보가 포함된 필드 삭제 (API 응답 구조에 따라 존재 가능)
        if 'diarization' in clean_seg:
            del clean_seg['diarization']
        
        # words 추출
        words = clean_seg.get("words", [])
        for word in words:
            # word는 [start_ms, end_ms, text] 형태
            if isinstance(word, list) and len(word) >= 3:
                word_start_ms = word[0]  # 밀리초
                word_end_ms = word[1]    # 밀리초
                word_text = word[2]      # 텍스트
                
                # 밀리초를 초로 변환
                word_start = word_start_ms / 1000.0
                word_end = word_end_ms / 1000.0
                
                if word_text and word_text.strip():
                    all_words.append({
                        "text": word_text.strip(),
                        "start": word_start,
                        "end": word_end
                    })
        
        final_segments.append(clean_seg)
    
    # duration 계산 (librosa 사용)
    y, sr = librosa.load(audio_file_path, sr=16000)
    duration = float(len(y) / sr)
    
    return {
        "text": full_text,
        "words": all_words,
        "segments": final_segments,
        "duration": duration
    }