from app import app as application  # EB needs 'application'

# Optional for local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=8000)