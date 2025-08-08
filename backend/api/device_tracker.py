from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from utils.auth_utils import decode_jwt_token
import psycopg2


router = APIRouter()

conn = psycopg2.connect(
    host="localhost",
    database="support_db",
    user="postgres",
    password="123456"
)
cursor = conn.cursor()

@app.post("/save-device-info")
async def save_device_info(data: DeviceInfo, request: Request):

    # ─── JWT Decode ───
    try:
        credentials = credentials.credentials  # Unpacking the tuple

        try:
            payload = decode_jwt_token(credentials)
        except ExpiredSignatureError:
            if not refresh_token:
                raise HTTPException(status_code=401, detail="Token expired. Provide refresh_token.")
            async with httpx.AsyncClient() as client:
                res = await client.post(REFR_URL, json={"refresh_token": refresh_token})
                if res.status_code != 200:
                    raise HTTPException(status_code=401, detail="Refresh token invalid")
                new_token = res.json()["token"]
                payload = decode_jwt_token(new_token)

        user_email = payload.get("sub", "anonymous")  # using email (string)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    cursor.execute(
        "INSERT INTO device_info (user_id, ip, os, browser, user_agent, screen_resolution, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user_email, data.ip, data.os, data.browser, data.userAgent, data.screenResolution, datetime.utcnow())
    )
    conn.commit()
    return {"message": "Device info saved in PostgreSQL"}
   

@app.get("/get-logs")
async def get_logs():
    return JSONResponse(content=device_logs)

# Already existing IP endpoint
@app.get("/get-ip")
async def get_ip(request: Request):
    ip_address = request.headers.get('X-Forwarded-For', request.client.host)
    if ip_address and "," in ip_address:
        ip_address = ip_address.split(",")[0].strip()
    return JSONResponse(content={"ip": ip_address})

