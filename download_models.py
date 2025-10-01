import gdown
import os

# Google Drive file IDs (make sure they are shared with "Anyone with the link")
CLASSIFIER_ID = "164XFIzeaSKaPSaDwrBAWSscU6_0martE"
REGRESSOR_ID = "16vnUZg4roLRwReUKrYFLk4igroBAGF8m"

# Local paths
os.makedirs("models", exist_ok=True)
classifier_path = "models/alpha_cash_classifier_200k.joblib"
regressor_path = "models/alpha_cash_regressor_200k.joblib"

def download_if_missing(file_id, out_path, name):
    if not os.path.exists(out_path):
        print(f"Downloading {name}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, out_path, quiet=False, fuzzy=True)
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
    else:
        print(f"{name} already exists, skipping download.")

download_if_missing(CLASSIFIER_ID, classifier_path, "classifier model")
download_if_missing(REGRESSOR_ID, regressor_path, "regressor model")

print("✅ Models ready!")
