import os
import sys
import time
import subprocess
import requests

API_URL = "http://127.0.0.1:5000/api/analyze"

def check_adb():
    try:
        res = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
        lines = [line.strip() for line in res.stdout.split("\n") if line.strip()]
        devices = [l.split("\t")[0] for l in lines[1:] if "\tdevice" in l]
        return devices
    except Exception:
        return []

def get_latest_sms_adb():
    cmd = [
        "adb", "shell", "content", "query",
        "--uri", "content://sms/inbox",
        "--projection", "address,body,date",
        "--sort", "date DESC"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = res.stdout
        # Parse the first row
        for line in output.splitlines():
            if "Row:" in line and "body=" in line:
                # Extract address, body
                parts = line.split(", ")
                address = "Unknown"
                body = ""
                for p in parts:
                    if p.startswith("address="):
                        address = p.replace("address=", "")
                    elif p.startswith("body="):
                        body = p.replace("body=", "")
                if body:
                    return address, body
        return None, None
    except Exception:
        return None, None

def monitor_live_phone():
    print("=" * 65)
    print("SCAMSHIELD AI - LIVE ANDROID USB SYNC MONITOR")
    print("=" * 65)
    print("Checking for connected Android phone via USB...")

    devices = check_adb()
    if not devices:
        print("\n[!] No Android device detected via ADB.")
        print("To connect your real Android phone directly:")
        print("1. Enable 'Developer Options' on your Android phone.")
        print("2. Turn ON 'USB Debugging'.")
        print("3. Connect phone to laptop with a USB cable.")
        print("4. Allow USB Debugging prompt on phone screen.")
        print("\nAlternatively, you can access ScamShield AI directly from your")
        print("phone's browser over Wi-Fi without needing USB cable!")
        return

    print(f"\n[OK] Connected Phone Detected: {devices[0]}")
    print("[*] Listening for incoming live SMS in real time (Ctrl+C to stop)...\n")

    last_seen_msg = None

    while True:
        sender, msg = get_latest_sms_adb()
        if msg and msg != last_seen_msg:
            last_seen_msg = msg
            print(f"\n[!] LIVE SMS ARRIVED on Phone from: {sender}")
            print(f"    Message: \"{msg}\"")
            print("    Scanning with ScamShield AI API...")

            try:
                r = requests.post(API_URL, json={"message": msg, "model_choice": "Multinomial Naive Bayes"})
                if r.status_code == 200:
                    data = r.json()
                    is_spam = data.get("is_spam", False)
                    score = data.get("analysis", {}).get("composite_risk_score", 0)
                    indicators = [i["name"] for i in data.get("analysis", {}).get("indicators", [])]

                    if is_spam:
                        print(f"    🚨 VERDICT: FRAUDULENT / PHISHING SMS DETECTED!")
                        print(f"    ⚠️ Risk Score: {score}%")
                        print(f"    🚩 Flags: {', '.join(indicators)}")
                    else:
                        print(f"    ✅ VERDICT: LIKELY SAFE (HAM)")
                        print(f"    Risk Score: {score}%")
            except Exception as e:
                print(f"    Error contacting API: {e}")

        time.sleep(2)

if __name__ == "__main__":
    monitor_live_phone()
