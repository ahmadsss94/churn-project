import joblib, json
art = joblib.load("churn_model.joblib")
feat = art["feature_order"]

# start with zeros; set strings for categorical if present
payload = {k: 0 for k in feat}
for k in ("last_level", "gender", "location"):
    if k in payload:
        payload[k] = "unknown"  # replace with real values later

print(json.dumps(payload, indent=2))
