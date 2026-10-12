import os
import json
import time

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "analysis_history.json")

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_analysis_record(record: dict):
    history = load_history()
    # Add timestamp and ID
    record["id"] = int(time.time() * 1000)
    record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    history.append(record)
    # Keep latest 100 entries
    if len(history) > 100:
        history = history[-100:]
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")
    return record

def clear_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception:
            pass

def get_dashboard_stats():
    history = load_history()
    total = len(history)
    if total == 0:
        return {
            "total_analyzed": 0,
            "ham_count": 0,
            "spam_count": 0,
            "risk_distribution": {"Low": 0, "Elevated": 0, "High": 0, "Severe": 0},
            "recent_analyses": [],
            "has_history": False
        }

    ham_count = sum(1 for h in history if h.get("prediction") == "Likely Safe")
    spam_count = sum(1 for h in history if h.get("prediction") == "Potential Spam")
    
    risk_distribution = {"Low": 0, "Elevated": 0, "High": 0, "Severe": 0}
    for h in history:
        lvl = h.get("risk_level", "Low")
        if lvl in risk_distribution:
            risk_distribution[lvl] += 1
        else:
            risk_distribution["Low"] += 1

    return {
        "total_analyzed": total,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "risk_distribution": risk_distribution,
        "recent_analyses": list(reversed(history[-10:])),
        "has_history": True
    }
