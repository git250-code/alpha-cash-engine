# download_models.py
import os
import gdown

# Google Drive IDs (ensure files set to "Anyone with the link -> Viewer")
CLASSIFIER_ID = "164XFIzeaSKaPSaDwrBAWSscU6_0martE"
REGRESSOR_ID = "16vnUZg4roLRwReUKrYFLk4igroBAGF8m"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "alpha_cash_classifier_200k.joblib")
REGRESSOR_PATH  = os.path.join(MODEL_DIR, "alpha_cash_regressor_200k.joblib")

def download_if_missing(file_id: str, out_path: str, name: str):
    if os.path.exists(out_path):
        print(f"[skip] {name} exists at {out_path}")
        return True
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download] {name} from {url} -> {out_path}")
    try:
        ok = gdown.download(url, out_path, quiet=False, fuzzy=True)
        if ok:
            print(f"[ok] downloaded {name}")
            return True
        else:
            print(f"[fail] gdown returned False for {name}")
            return False
    except Exception as e:
        print(f"[error] failed to download {name}: {e}")
        return False

def ensure_models():
    ok1 = download_if_missing(CLASSIFIER_ID, CLASSIFIER_PATH, "classifier")
    ok2 = download_if_missing(REGRESSOR_ID, REGRESSOR_PATH, "regressor")
    if not (ok1 and ok2):
        raise RuntimeError("Model download failed. Check Google Drive sharing and quotas.")
    return CLASSIFIER_PATH, REGRESSOR_PATH

if __name__ == "__main__":
    ensure_models()
    print("Models ready.")
