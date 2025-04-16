import uvicorn

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)