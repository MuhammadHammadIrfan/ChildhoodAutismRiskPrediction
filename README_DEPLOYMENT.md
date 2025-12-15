# Streamlit Deployment Guide

## 🚀 Local Deployment

### Step 1: Train and Save the Model

```bash
# From project root directory
python src/model/train_and_save_model.py
```

This will:
- Train all 6 baseline models
- Select the best performing model
- Save the model to `models/best_model.pkl`

### Step 2: Run Streamlit App Locally

```bash
# From project root directory
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 3: Test the Application

1. Answer all 10 behavioral questions
2. Fill in demographic information
3. Click "Predict Autism Risk"
4. Review the results and recommendations

---

## ☁️ Streamlit Cloud Deployment

### Prerequisites

1. **GitHub Repository**: Your code must be in a GitHub repository (✅ Already done)
2. **Streamlit Cloud Account**: Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)

### Step 1: Prepare for Deployment

Ensure these files are in your repository:
- ✅ `app/app.py` - Main application
- ✅ `app/requirements.txt` - Dependencies
- ✅ `models/best_model.pkl` - Trained model (you'll need to commit this)
- ✅ `.streamlit/config.toml` - Configuration

### Step 2: Commit the Trained Model

```bash
# Train the model first
python src/model/train_and_save_model.py

# Add and commit the model file
git add models/best_model.pkl
git add app/app.py app/requirements.txt
git add .streamlit/config.toml
git add src/model/train_and_save_model.py
git commit -m "Add Streamlit app and trained model for deployment"
git push origin main
```

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io)

2. **Sign in with GitHub**

3. **Create New App**:
   - Click "New app"
   - Select your repository: `ChildhoodAutismRiskPrediction`
   - Branch: `main`
   - Main file path: `app/app.py`
   - Click "Deploy"

4. **Wait for Deployment**:
   - Streamlit Cloud will install dependencies
   - Build and deploy your app
   - Takes ~2-5 minutes

5. **Get Your Public URL**:
   - Format: `https://[your-app-name].streamlit.app`
   - Share this link with anyone!

### Step 4: Update Your App

When you make changes:

```bash
git add .
git commit -m "Update app features"
git push origin main
```

Streamlit Cloud will automatically redeploy your app! 🎉

---

## 🔧 Troubleshooting

### Model File Too Large for Git

If the model file is too large (>100MB), use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track the model file
git lfs track "models/*.pkl"
git add .gitattributes
git add models/best_model.pkl
git commit -m "Add model with Git LFS"
git push origin main
```

### Missing Dependencies

If the app crashes on Streamlit Cloud:
1. Check the logs in Streamlit Cloud dashboard
2. Ensure all packages are listed in `app/requirements.txt`
3. Pin specific versions for reproducibility

### Model Not Loading

If you see "Model file not found":
1. Ensure `models/best_model.pkl` is committed to Git
2. Check the file path in `app.py` is correct
3. Verify the model was pushed to GitHub

---

## 📊 App Features

### Current Features:
- ✅ 10 behavioral screening questions (A1-A10)
- ✅ Demographic information collection
- ✅ Real-time risk prediction
- ✅ Probability scores
- ✅ Detailed interpretations
- ✅ Professional styling and UX

### Future Enhancements:
- 📈 Add model performance metrics
- 📊 Add feature importance visualization
- 🌐 Multi-language support
- 📱 Mobile-responsive design improvements
- 📧 Email results option
- 📥 PDF report generation

---

## 🐳 Docker Deployment (Next Step)

After Streamlit Cloud deployment is working, we'll containerize with Docker:

```dockerfile
# Preview: Dockerfile (coming next)
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r app/requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py"]
```

This will make deployment even more flexible! 🚀

---

## 📝 Notes

- **Data Privacy**: No user data is stored or logged
- **Model Updates**: Retrain and commit new model to update predictions
- **Monitoring**: Check Streamlit Cloud analytics for usage stats
- **Cost**: Streamlit Cloud has a free tier for public apps

---

## 🎯 Deployment Checklist

- [ ] Train model: `python src/model/train_and_save_model.py`
- [ ] Test locally: `streamlit run app/app.py`
- [ ] Commit model to Git
- [ ] Push to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] Test deployed app
- [ ] Share public URL

**Happy Deploying! 🎉**
