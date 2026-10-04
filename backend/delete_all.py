import requests

res = requests.get("http://127.0.0.1:8000/api/v1/documents")
if res.status_code == 200:
    docs = res.json()
    for d in docs:
        doc_id = d.get("id")
        print(f"Deleting {doc_id}")
        requests.delete(f"http://127.0.0.1:8000/api/v1/documents/{doc_id}")
print("Done")

