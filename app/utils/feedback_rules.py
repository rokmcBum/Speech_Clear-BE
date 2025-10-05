# app/utils/feedback_rules.py
def make_feedback(m: dict) -> str:
    msgs = []
    if m.get("rate_wpm", 0) < 100:
        msgs.append("말속도를 조금 더 올려보세요.")
    if (m.get("pitch_mean_hz") or 0) < 110:
        msgs.append("톤을 조금 더 높여보세요.")
    if (m.get("pause_ratio") or 0) > 0.3:
        msgs.append("침묵이 많습니다. 끊김 없이 말해보세요.")
    if not msgs:
        return "좋습니다. 현재 톤과 템포를 유지하세요."
    return " ".join(msgs)
