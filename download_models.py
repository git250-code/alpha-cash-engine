import gdown
import os

# Google Drive file IDs
CLASSIFIER_ID = "164XFIzeaSKaPSaDwrBAWSscU6_0martE"
REGRESSOR_ID = "16vnUZg4roLRwReUKrYFLk4igroBAGF8m"

os.makedirs("models", exist_ok=True)

classifier_path = "models/alpha_cash_classifier_200k.joblib"
regressor_path = "models/alpha_cash_regressor_200k.joblib"

if not os.path.exists(classifier_path):
    print("Downloading classifier model...")
    gdown.download(f"https://drive.google.com/uc?id={CLASSIFIER_ID}", classifier_path, quiet=False)

if not os.path.exists(regressor_path):
    print("Downloading regressor model...")
    gdown.download(f"https://drive.google.com/uc?id={REGRESSOR_ID}", regressor_path, quiet=False)

print("Models ready!")
