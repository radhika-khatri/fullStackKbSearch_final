# MongoDB Setup Instructions

## Issue
The current MongoDB Atlas credentials are invalid, causing authentication failures and preventing the feedback system from working.

## Solutions

### Option 1: Fix MongoDB Atlas Credentials (Recommended)
1. Log into your MongoDB Atlas account
2. Go to Database Access and create/update the user credentials
3. Update the `.env` file with correct credentials:
   ```
   MONGO_URI=mongodb+srv://YOUR_USERNAME:YOUR_PASSWORD@cluster0.tpatz5r.mongodb.net/chat_support?retryWrites=true&w=majority
   ```

### Option 2: Use Local MongoDB (Development)
1. Install MongoDB locally:
   ```bash
   # Ubuntu/Debian
   sudo apt install mongodb-community
   
   # macOS
   brew install mongodb-community
   ```
2. Start MongoDB service:
   ```bash
   sudo systemctl start mongod
   ```
3. Update `.env` file:
   ```
   MONGO_URI=mongodb://localhost:27017/chat_support
   ```

### Option 3: Use SQLite Fallback (Quick Fix)
I can implement a SQLite fallback for development purposes.

## Current Error
```
bad auth : authentication failed
```

This means the username/password combination in the MongoDB URI is incorrect.

## Next Steps
Please choose one of the options above, or provide the correct MongoDB credentials.
