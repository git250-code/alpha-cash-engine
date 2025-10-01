### POST /ml_predict_upload

Accepts **multipart/form-data** with:

- **file**: optional CSV (text/csv) or image (image/png|jpeg) containing key: value rows  
- **form fields**: any of the 13 inputs (`current_bank_balance`, `monthly_expense`, …)

Extraction priority:  
1. CSV → first-row columns  
2. OCR on image → lines `key: value`  
3. Manual form values override

Returns:
```json
{
  "suggested_action": "<str>",
  "expected_return": <float>
}
