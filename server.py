from core.app import create_app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", port=8000)
