import requests
import json 
response = requests.post(
    "http://localhost:8000/check-compliance",
    json={
        "policy_url": "https://docs.stripe.com/treasury/marketing-treasury",
        "content_url": "https://mercury.com/"
    }
)
json_obj = response.json()
formatted_data = json.dumps(json_obj, indent=4)
print(formatted_data)