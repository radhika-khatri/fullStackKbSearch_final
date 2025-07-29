import subprocess
import time
import requests
import re

# === CONFIG ===
PORT = 8000  # Your FastAPI port
UVICORN_CMD = f"uvicorn main:app --reload --port {PORT}"  # Change `main:app` to your FastAPI entrypoint
ngrok_path = "./frontend/node_modules/.bin/ngrok"
def start_fastapi():
    print(" Starting FastAPI server...")
    return subprocess.Popen(UVICORN_CMD, shell=True)

def start_ngrok(port):
    print(" Starting ngrok tunnel...")
    return subprocess.Popen(f"{ngrok_path} http {port}", shell=True, stdout=subprocess.DEVNULL)

def get_ngrok_url():
    time.sleep(3)  # Give ngrok a few seconds to start
    try:
        res = requests.get("http://localhost:4040/api/tunnels")
        tunnels = res.json().get("tunnels", [])
        for tunnel in tunnels:
            if tunnel["proto"] == "https":
                return tunnel["public_url"]
    except Exception as e:
        print(" Failed to fetch ngrok URL:", e)
    return None

def main():
    fastapi_proc = start_fastapi()
    ngrok_proc = start_ngrok(PORT)

    ngrok_url = None
    for _ in range(10):
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            break
        time.sleep(1)

    if ngrok_url:
        print("\n FastAPI + Ngrok are up and running!")
        print(f" Use this Callback URL in Meta Webhook settings: {ngrok_url}/webhook")
    else:
        print("Could not detect ngrok tunnel. Please check manually.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Shutting down...")
        fastapi_proc.terminate()
        ngrok_proc.terminate()

if __name__ == "__main__":
    main()
