import os
import urllib.request
import zipfile
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "sms_spam_collection.csv")

def ensure_dataset():
    """
    Downloads and caches the SMS Spam Collection dataset.
    Returns a pandas DataFrame with columns: ['label', 'message']
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)

    raw_tsv_path = os.path.join(DATA_DIR, "SMSSpamCollection")
    
    # Try downloading raw tsv or zip
    urls = [
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
        "https://raw.githubusercontent.com/stefan-bergstein/SMS-Spam-Collection/master/SMSSpamCollection",
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    ]

    downloaded = False
    for url in urls:
        try:
            print(f"Attempting to download dataset from: {url}")
            temp_dest = os.path.join(DATA_DIR, "temp_download")
            urllib.request.urlretrieve(url, temp_dest)
            
            if url.endswith(".zip"):
                with zipfile.ZipFile(temp_dest, 'r') as zf:
                    zf.extract("SMSSpamCollection", DATA_DIR)
                if os.path.exists(temp_dest):
                    os.remove(temp_dest)
                if os.path.exists(raw_tsv_path):
                    downloaded = True
                    break
            else:
                # TSV file
                os.replace(temp_dest, raw_tsv_path)
                downloaded = True
                break
        except Exception as e:
            print(f"Failed from {url}: {e}")
            continue

    if not downloaded:
        raise RuntimeError("Unable to download dataset from mirrors. Please check internet connection.")

    # Parse dataset (tab-separated: label, message)
    df = pd.read_csv(
        raw_tsv_path,
        sep='\t',
        header=None,
        names=['label', 'message'],
        encoding='utf-8',
        quoting=3, # csv.QUOTE_NONE
        on_bad_lines='skip'
    )

    # Clean leading/trailing spaces
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['message'] = df['message'].astype(str).str.strip()

    # Filter to only valid ham and spam
    df = df[df['label'].isin(['ham', 'spam'])].copy()
    df = df[df['message'] != '']

    # Save cleaned csv
    df.to_csv(DATA_FILE, index=False)
    print(f"Dataset successfully prepared: {len(df)} samples ({sum(df['label']=='ham')} ham, {sum(df['label']=='spam')} spam)")
    return df

if __name__ == "__main__":
    df = ensure_dataset()
    print(df.head())
    print(df['label'].value_counts())
