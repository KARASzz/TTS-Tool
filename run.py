from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=7860,
        reload=True,
        reload_dirs=[str(ROOT)],
        reload_excludes=["outputs/*", "uploads/*", "*.pyc", "__pycache__/*"],
    )
