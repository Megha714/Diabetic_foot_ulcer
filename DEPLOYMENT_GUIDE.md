# Deploy to Vercel - Step by Step Guide

## ⚠️ Important Notes

**Vercel has limitations for Flask apps:**
- **File uploads** may not work properly (your foot image detection)
- **Background processes** like ML model loading may timeout
- **Better alternatives**: Render, Railway, or PythonAnywhere

## If you still want to try Vercel:

### Prerequisites
1. Create a Vercel account at https://vercel.com
2. Install Vercel CLI (optional)

### Method 1: Deploy via GitHub (Recommended)

1. **Push your code to GitHub** (Already done ✅)

2. **Go to Vercel Dashboard**:
   - Visit https://vercel.com/dashboard
   - Click "Add New" → "Project"

3. **Import your GitHub repository**:
   - Select "Import Git Repository"
   - Choose `Megha714/Diabetic_foot_ulcer`
   - Click "Import"

4. **Configure Environment Variables**:
   Click "Environment Variables" and add:
   ```
   MONGO_URI=mongodb+srv://1by22cs099_db_user:lHyJMo6qgN3U1tLQ@dfu-cluster.3cbzdtc.mongodb.net/?retryWrites=true&w=majority
   SECRET_KEY=your-secret-key-change-this-in-production-2024
   FLASK_ENV=production
   ```

5. **Deploy**:
   - Click "Deploy"
   - Wait for deployment to complete
   - You'll get a URL like: `https://diabetic-foot-ulcer.vercel.app`

### Method 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
cd "D:\dfu_project 2\dfu_project"
vercel --prod
```

Follow the prompts and add environment variables when asked.

---

## 🚀 Better Alternatives for Flask + ML Apps

### 1. **Render** (Recommended) ⭐

**Why**: Free tier, supports ML models, file uploads, background workers

**Steps**:
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect `Megha714/Diabetic_foot_ulcer`
5. Configure:
   - **Name**: dfu-detection
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Instance Type**: Free
6. Add Environment Variables (same as above)
7. Click "Create Web Service"

### 2. **Railway** (Easy & Fast)

**Steps**:
1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `Megha714/Diabetic_foot_ulcer`
5. Add environment variables
6. Deploy automatically starts!

### 3. **PythonAnywhere** (Python-Specific Hosting)

**Steps**:
1. Go to https://www.pythonanywhere.com
2. Create free account
3. Upload your code or clone from GitHub
4. Configure WSGI file
5. Set environment variables
6. Start web app

---

## 📝 Important Changes Needed for Production

### 1. Update `app.py` for production:

Add at the bottom of `app.py`:
```python
if __name__ == '__main__':
    # For local development
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # For production (Vercel/Render/Railway)
    # WSGI server will handle this
    pass
```

### 2. Add to `.gitignore`:
```
.env
*.pyc
__pycache__/
uploads/
checkpoints/best_model.pth
.vercel
```

### 3. Create `.env` file (DON'T COMMIT THIS):
```
MONGO_URI=mongodb+srv://1by22cs099_db_user:lHyJMo6qgN3U1tLQ@dfu-cluster.3cbzdtc.mongodb.net/?retryWrites=true&w=majority
SECRET_KEY=your-secret-key-change-this-in-production-2024
```

### 4. Update `app.py` to use environment variables:
```python
import os
from dotenv import load_dotenv

load_dotenv()

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key')
```

---

## ⚠️ Vercel Limitations for Your App

1. **Model File Size**: Your `best_model.pth` is likely too large (>250MB limit)
2. **Execution Timeout**: 10 seconds for Hobby plan (ML inference may timeout)
3. **File Uploads**: Ephemeral filesystem (uploaded images disappear)
4. **Memory**: Limited to 1024MB (your model may need more)

## ✅ Recommended Solution

**Use Render** for your DFU Detection app because:
- ✅ Supports large ML models
- ✅ Persistent file storage for uploads
- ✅ No timeout issues
- ✅ Free tier available
- ✅ Easy MongoDB integration
- ✅ Better for Flask + PyTorch apps

---

## 🔗 Quick Deploy to Render

1. Push your code to GitHub ✅ (Already done!)
2. Go to: https://dashboard.render.com/
3. Click "New +" → "Web Service"
4. Connect GitHub → Select your repo
5. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Add `gunicorn` to `requirements.txt`
7. Add environment variables
8. Click "Create Web Service"

**Done!** Your app will be live at `https://your-app-name.onrender.com`

---

## 📞 Need Help?

- Render Docs: https://render.com/docs/deploy-flask
- Railway Docs: https://docs.railway.app/deploy/deployments
- MongoDB Atlas: Already configured ✅
