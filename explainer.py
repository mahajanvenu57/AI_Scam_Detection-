import re

# 1. Urgent language
URGENCY_KEYWORDS = [
    r"\burgent\b", r"\bimmediately\b", r"\bexpire[s]?\b", r"\bexpiring\b",
    r"\blimited time\b", r"\bwithin 24\b", r"\bwithin 48\b", r"\bhurry\b",
    r"\blast chance\b", r"\bact now\b", r"\binstant\b", r"\bdeadline\b"
]

# 2. Pressure / Threat language
THREAT_KEYWORDS = [
    r"\baction required\b", r"\bsuspended\b", r"\bblocked\b", r"\bdeactivated\b",
    r"\bwarning\b", r"\bpenalty\b", r"\barrest\b", r"\blegal action\b",
    r"\bdisconnected\b", r"\bterminated\b", r"\bfinal notice\b", r"\bimmediate action\b",
    r"\bunauthorized\b", r"\bdisconnection\b"
]

# 3. Prize / Reward language
REWARD_KEYWORDS = [
    r"\bwinner\b", r"\bwon\b", r"\bcongratulations\b", r"\bcongrats\b",
    r"\bprize\b", r"\blottery\b", r"\bclaim\b", r"\bjackpot\b",
    r"\bselected\b", r"\breward\b", r"\bgift card\b", r"\bfree bonus\b",
    r"\bcash prize\b", r"\b100% free\b", r"\bexclusive offer\b", r"\bgift voucher\b",
    r"\blucky draw\b", r"\bfree\b"
]

# 4. Financial request
FINANCIAL_KEYWORDS = [
    r"\bbank\b", r"\baccount\b", r"\bdebit\b", r"\bcredit\b", r"\brefund\b",
    r"\bcashback\b", r"\bloan\b", r"\bpre-approved\b", r"\btransaction\b",
    r"\btransfer\b", r"\bbilling\b", r"\binvoice\b", r"\bpayment\b",
    r"\bfee\b", r"\bdeposit\b", r"\bwire\b"
]

# 5. Credential / Password request
CREDENTIAL_KEYWORDS = [
    r"\bkyc\b", r"\bpan\b", r"\baadhaar\b", r"\bpin\b", r"\botp\b",
    r"\bpassword\b", r"\bcvv\b", r"\bverify\b", r"\bverification\b",
    r"\blogin\b", r"\bsecurity code\b", r"\bcredentials\b", r"\bpasscode\b"
]

# 6. Suspicious general keywords
SUSPICIOUS_KEYWORDS = [
    r"\bclick here\b", r"\btap here\b", r"\bsubscribe\b", r"\bunsubscribe\b",
    r"\btxt\b", r"\brply\b", r"\bstop2\b", r"\brated\b", r"\bper week\b",
    r"\bp/min\b", r"\bcost\b"
]

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|org|net|xyz|top|ru|io|ly|co|info|biz|tk|ml|ga|cf|gq|club|online|site)/?\S*)", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{5,6}\b|\b[6-9]\d{9}\b")
CURRENCY_REGEX = re.compile(r"([$£€₹]|rs\.?|inr|usd|gbp)\s?\d+", re.IGNORECASE)

DISCLAIMER_TEXT = (
    "This tool provides an AI-assisted risk assessment. It does not verify the sender’s identity "
    "or guarantee that a message is fraudulent or legitimate. These are warning indicators only and "
    "must never be described as absolute proof of fraud."
)

def extract_features(text: str) -> dict:
    text = str(text) if text is not None else ""
    char_count = len(text)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    digit_count = sum(c.isdigit() for c in text)
    punct_count = sum(c in "!?,.:;\"'()[]{}-_@#$%^&*~`" for c in text)
    upper_chars = sum(c.isupper() for c in text)
    alpha_chars = sum(c.isalpha() for c in text)
    uppercase_ratio = (upper_chars / alpha_chars) if alpha_chars > 0 else 0.0
    digit_ratio = (digit_count / char_count) if char_count > 0 else 0.0

    urls = URL_REGEX.findall(text)
    phones = PHONE_REGEX.findall(text)
    currencies = CURRENCY_REGEX.findall(text)

    def find_matches(patterns):
        matches = []
        for p in patterns:
            found = re.findall(p, text, re.IGNORECASE)
            if found:
                matches.extend(found)
        return list(set(m.lower() for m in matches))

    urgency_matches = find_matches(URGENCY_KEYWORDS)
    threat_matches = find_matches(THREAT_KEYWORDS)
    reward_matches = find_matches(REWARD_KEYWORDS)
    financial_matches = find_matches(FINANCIAL_KEYWORDS)
    credential_matches = find_matches(CREDENTIAL_KEYWORDS)
    suspicious_matches = find_matches(SUSPICIOUS_KEYWORDS)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "digit_count": digit_count,
        "digit_ratio": round(digit_ratio, 3),
        "punct_count": punct_count,
        "uppercase_ratio": round(uppercase_ratio, 3),
        "url_count": len(urls),
        "phone_count": len(phones),
        "currency_count": len(currencies),
        "urgency_matches": urgency_matches,
        "threat_matches": threat_matches,
        "reward_matches": reward_matches,
        "financial_matches": financial_matches,
        "credential_matches": credential_matches,
        "suspicious_matches": suspicious_matches,
        "detected_urls": urls,
    }

def analyze_caution_indicators(text: str, ml_probability: float = None) -> dict:
    feat = extract_features(text)
    indicators = []
    highlight_terms = set()

    # 1. Urgent language
    if feat["urgency_matches"]:
        indicators.append({
            "name": "Urgent Language",
            "category": "Time Pressure / Urgency",
            "level": "HIGH",
            "icon": "⏳",
            "description": f"Urgency keywords detected: {', '.join(feat['urgency_matches'])}. Scammers create artificial panic to bypass critical scrutiny."
        })
        highlight_terms.update(feat["urgency_matches"])

    # 2. Pressure / Threat language
    if feat["threat_matches"]:
        indicators.append({
            "name": "Pressure / Threat Language",
            "category": "Coercion & Penalty",
            "level": "HIGH",
            "icon": "⚠️",
            "description": f"Coercive/threat terms detected: {', '.join(feat['threat_matches'])}. Unsubstantiated threats of suspension or legal action are typical social engineering tactics."
        })
        highlight_terms.update(feat["threat_matches"])

    # 3. Prize / Reward language
    if feat["reward_matches"]:
        indicators.append({
            "name": "Prize / Reward Language",
            "category": "Unrealistic Incentive",
            "level": "HIGH",
            "icon": "🎁",
            "description": f"Prize/lure keywords detected: {', '.join(feat['reward_matches'])}. Unsolicited reward or lottery announcements are primary smishing baits."
        })
        highlight_terms.update(feat["reward_matches"])

    # 4. Financial request
    if feat["financial_matches"]:
        indicators.append({
            "name": "Financial Request",
            "category": "Monetary / Banking Bait",
            "level": "HIGH",
            "icon": "💳",
            "description": f"Financial keywords detected: {', '.join(feat['financial_matches'])}. Messages discussing accounts, transfers, or pre-approved loans require independent bank verification."
        })
        highlight_terms.update(feat["financial_matches"])

    # 5. Credential / Password request
    if feat["credential_matches"]:
        indicators.append({
            "name": "Credential / Password Request",
            "category": "Identity & Security Harvest",
            "level": "CRITICAL",
            "icon": "🔐",
            "description": f"Sensitive credential triggers detected: {', '.join(feat['credential_matches'])}. Legitimate institutions never request OTP, PIN, password, or KYC submission via unverified SMS."
        })
        highlight_terms.update(feat["credential_matches"])

    # 6. Suspicious URL
    if feat["url_count"] > 0:
        indicators.append({
            "name": "Suspicious URL",
            "category": "Smishing / Web Link",
            "level": "CRITICAL",
            "icon": "🔗",
            "description": f"Found {feat['url_count']} link(s) embedded in text. Malicious SMS campaigns direct recipients to fraudulent phishing portals."
        })
        for u in feat["detected_urls"]:
            highlight_terms.add(u)

    # 7. Currency symbols
    if feat["currency_count"] > 0:
        indicators.append({
            "name": "Currency Symbols",
            "category": "Monetary Quantifiers",
            "level": "MEDIUM",
            "icon": "💰",
            "description": "Detected currency denominations ($, GBP, EUR, INR, Rs) conveying monetary amounts or payout promises."
        })

    # 8. Phone-number-like patterns
    if feat["phone_count"] > 0:
        indicators.append({
            "name": "Phone-number-like Patterns",
            "category": "Direct Call / Shortcode",
            "level": "MEDIUM",
            "icon": "📞",
            "description": "Detected phone numbers or premium-rate shortcodes encouraging unverified direct calls or replies."
        })

    # 9. Excessive numbers
    if feat["digit_ratio"] > 0.15 and feat["digit_count"] >= 5:
        indicators.append({
            "name": "Excessive Numbers",
            "category": "Numerical Density",
            "level": "MEDIUM",
            "icon": "🔢",
            "description": f"High density of numeric digits ({feat['digit_count']} digits, {int(feat['digit_ratio']*100)}% of content) commonly seen in fake transaction IDs and serial codes."
        })

    # 10. Suspicious keywords / Shouting text
    if feat["suspicious_matches"] or feat["uppercase_ratio"] > 0.25:
        details_list = []
        if feat["suspicious_matches"]:
            details_list.append(f"Keywords: {', '.join(feat['suspicious_matches'])}")
            highlight_terms.update(feat["suspicious_matches"])
        if feat["uppercase_ratio"] > 0.25:
            details_list.append(f"Capitalization ratio: {int(feat['uppercase_ratio']*100)}%")
        
        indicators.append({
            "name": "Suspicious Keywords & Format",
            "category": "Formatting Anomalies",
            "level": "MEDIUM",
            "icon": "🔠",
            "description": f"Detected formatting anomalies: {'; '.join(details_list)}."
        })

    # Heuristic scoring
    heuristic_score = 0
    if feat["url_count"] > 0:
        heuristic_score += 35
    if feat["credential_matches"]:
        heuristic_score += 25
    if feat["reward_matches"]:
        heuristic_score += 25
    if feat["financial_matches"]:
        heuristic_score += 20
    if feat["urgency_matches"] or feat["threat_matches"]:
        heuristic_score += 20
    if feat["currency_count"] > 0:
        heuristic_score += 10
    if feat["phone_count"] > 0:
        heuristic_score += 10
    heuristic_score = min(heuristic_score, 100)

    # Composite risk score
    if ml_probability is not None:
        composite_score = round((ml_probability * 70.0) + (heuristic_score * 0.30), 1)
    else:
        composite_score = float(heuristic_score)
    composite_score = min(max(composite_score, 0.0), 100.0)

    # Risk Level mapping
    if composite_score >= 75:
        risk_level = "Severe"
    elif composite_score >= 50:
        risk_level = "High"
    elif composite_score >= 25:
        risk_level = "Elevated"
    else:
        risk_level = "Low"

    # Actionable safety guidance
    safety_guidance = []
    if feat["credential_matches"] or feat["financial_matches"]:
        safety_guidance.append("Never share OTP, PIN, passwords, or banking credentials under any circumstances.")
        safety_guidance.append("Banks and legitimate institutions will never demand confidential verification over SMS.")
    if feat["url_count"] > 0:
        safety_guidance.append("Do NOT tap or open embedded links from unverified senders. Manually type the official domain in your browser.")
    if feat["reward_matches"]:
        safety_guidance.append("Verify if you participated in any legitimate contest. Unsolicited lottery notifications are standard fraud lures.")
    if not safety_guidance:
        safety_guidance.append("Always exercise caution with unsolicited messages from unverified or unknown senders.")

    return {
        "features": feat,
        "indicators": indicators,
        "heuristic_score": heuristic_score,
        "composite_risk_score": composite_score,
        "risk_level": risk_level,
        "highlight_terms": list(highlight_terms),
        "safety_guidance": safety_guidance,
        "disclaimer": DISCLAIMER_TEXT
    }
