def make_feedback(word_metrics: list) -> str:
    feedbacks = ""
    for w in word_metrics:
        word = w["text"]
        m = w["metrics"]
        msgs = ""

        if m.get("dB", 0) < -25:  # 너무 작은 소리
            msgs += "발음이 너무 작습니다. 크게 말해보세요. "
        if m.get("pitch_mean_hz", 0) < 110:  # 피치 낮음
            msgs += "톤을 조금 더 올려보세요. "
        if m.get("duration_sec", 0) > 1.0:  # 발음이 너무 김
            msgs += "발음을 좀 더 짧게, 자연스럽게 말해보세요. "

        if msgs:
            feedbacks += "'" + word + "' 부분에서 " + msgs

    return feedbacks
