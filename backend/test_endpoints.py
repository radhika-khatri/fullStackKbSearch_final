#!/usr/bin/env python3
"""
Quick test script to check CORS and endpoint functionality
"""

import requests
import json

BASE_URL = "http://localhost:8004"

def test_cors_headers():
    """Test CORS preflight request"""
    print("Testing CORS headers...")
    try:
        response = requests.options(
            f"{BASE_URL}/api/qa/static-chat",
            headers={
                "Origin": "http://localhost:3001",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        print(f"OPTIONS response status: {response.status_code}")
        print(f"CORS headers: {dict(response.headers)}")
        return response.status_code == 200
    except Exception as e:
        print(f"CORS test failed: {e}")
        return False

def test_static_chat():
    """Test static chat endpoint"""
    print("\nTesting static chat endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/qa/static-chat",
            json={"question": "Hello, this is a test message"},
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3001"
            }
        )
        print(f"Static chat response status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Static chat test failed: {e}")
        return False

def test_multimodal_endpoint_health():
    """Test multimodal endpoint with text only"""
    print("\nTesting multimodal endpoint (text only)...")
    try:
        # Test with form data (text only)
        response = requests.post(
            f"{BASE_URL}/api/qa/chat-multimodal",
            data={"question": "What is AI?"},
            headers={"Origin": "http://localhost:3001"}
        )
        print(f"Multimodal response status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Multimodal test failed: {e}")
        return False

def test_image_health():
    """Test image processing health endpoint"""
    print("\nTesting image processing health...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/image/health",
            headers={"Origin": "http://localhost:3001"}
        )
        print(f"Image health response status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Image health test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing FastAPI endpoints and CORS configuration...\n")
    
    tests = [
        ("CORS Headers", test_cors_headers),
        ("Static Chat", test_static_chat),
        ("Multimodal Endpoint", test_multimodal_endpoint_health),
        ("Image Health", test_image_health)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results[test_name] = False
        print("-" * 50)
    
    print("\n📊 Test Results Summary:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'🎉 All tests passed!' if all_passed else '⚠️  Some tests failed'}")
