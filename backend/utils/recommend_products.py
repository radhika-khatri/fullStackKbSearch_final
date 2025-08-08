# recommend_products.py

from datetime import datetime
import psycopg2

# ─── PostgreSQL Connection ───
conn = psycopg2.connect(
    host="localhost",
    database="support_db",
    user="postgres",
    password="123456"
)
cursor = conn.cursor()

# ─── Example Cost Mapping ───
def classify_device_tier(os: str, browser: str, screen_resolution: str, user_agent: str) -> str:
    """Classify the device into Premium, Mid, or Budget tier based on parameters."""
    high_end_os = ["iOS", "macOS", "Windows 11", "Android 14"]
    premium_browsers = ["Safari", "Chrome"]
    high_resolutions = ["2560x1440", "3840x2160", "2880x1800"]

    if os in high_end_os or browser in premium_browsers or screen_resolution in high_resolutions:
        return "premium"
    elif "Samsung" in user_agent or "Pixel" in user_agent:
        return "mid"
    else:
        return "budget"

# ─── Recommend Products ───
def recommend_products(user_email: str, os: str, browser: str, user_agent: str, screen_resolution: str):
    """Return recommended products based on device tier and preferences."""
    tier = classify_device_tier(os, browser, screen_resolution, user_agent)
    cursor.execute("""
        SELECT name, price, category
        FROM products
        WHERE tier = %s
        ORDER BY price DESC
        LIMIT 10
    """, (tier,))

    products = cursor.fetchall()
    return {
        "tier": tier,
        "recommendations": [
            {"name": p[0], "price": p[1], "category": p[2]} for p in products
        ]
    }
