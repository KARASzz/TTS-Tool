from __future__ import annotations

import base64
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
load_dotenv(ROOT / ".env")

app = FastAPI(title="双引擎语音合成工具")

MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1").rstrip("/")
VOLC_TTS_URL = os.getenv("VOLC_TTS_URL", "https://openspeech.bytedance.com/api/v3/plan/tts/unidirectional")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com").rstrip("/")
MIMO_VOICES = [{"id": x, "name": n} for x, n in [("mimo_default", "MiMo 默认"), ("冰糖", "冰糖"), ("茉莉", "茉莉"), ("苏打", "苏打"), ("白桦", "白桦"), ("Mia", "Mia"), ("Chloe", "Chloe"), ("Milo", "Milo"), ("Dean", "Dean")]]
VOLC_VOICES = [{"id": x, "name": n} for x, n in [("zh_female_vv_uranus_bigtts", "通用女声"), ("zh_male_vv_uranus_bigtts", "通用男声"), ("zh_female_cancan", "灿灿"), ("zh_male_xiaotian", "小天")]]


def clean_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", Path(value).name)[:100]


def split_text(text: str, limit: int = 48000) -> list[str]:
    """迁移自 MiniMax-agent：优先按段落切分，避免超出 TTS 文本限制。"""
    text = text.strip()
    if len(text) <= limit:
        return [text]
    chunks, rest = [], text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest.strip())
            break
        cut = rest.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return [c for c in chunks if c]


async def mimo_tts(text: str, voice: str, style: str) -> bytes:
    key = os.getenv("MIMO_API_KEY")
    if not key:
        raise HTTPException(503, "未配置 MIMO_API_KEY")
    messages: list[dict[str, str]] = []
    if style.strip():
        messages.append({"role": "user", "content": style.strip()})
    messages.append({"role": "assistant", "content": text})
    payload = {"model": "mimo-v2.5-tts", "messages": messages, "audio": {"format": "wav", "voice": voice}}
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(f"{MIMO_BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
    if response.is_error:
        raise HTTPException(response.status_code, f"MiMo 请求失败: {response.text[:500]}")
    try:
        data = response.json()
        encoded = data["choices"][0]["message"]["audio"]["data"]
        return base64.b64decode(encoded)
    except (KeyError, IndexError, ValueError) as exc:
        raise HTTPException(502, f"MiMo 返回格式异常: {response.text[:500]}") from exc


async def volc_tts(text: str, voice: str, style: str) -> bytes:
    api_key = os.getenv("VOLC_API_KEY")
    if not api_key:
        raise HTTPException(503, "未配置 VOLC_API_KEY")
    request_id = str(uuid.uuid4())
    headers = {
        "Content-Type": "application/json",
        "X-Api-Resource-Id": os.getenv("VOLC_TTS_RESOURCE_ID", "seed-tts-2.0"),
        "X-Api-Request-Id": request_id,
    }
    headers["X-Api-Key"] = api_key
    payload: dict[str, Any] = {"user": {"uid": "local-user"}, "req_params": {"text": text, "speaker": voice or os.getenv("VOLC_VOICE", "")}}
    if style.strip():
        payload["req_params"]["text_type"] = "plain"
        payload["req_params"]["additions"] = style.strip()
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(VOLC_TTS_URL, headers=headers, json=payload)
    if response.is_error:
        raise HTTPException(response.status_code, f"火山方舟请求失败: {response.text[:500]}")
    content_type = response.headers.get("content-type", "")
    if "audio" in content_type or response.content[:4] == b"RIFF":
        return response.content
    try:
        data = response.json()
        encoded = data.get("data") or data.get("audio") or data.get("resp", {}).get("audio")
        if isinstance(encoded, dict):
            encoded = encoded.get("data")
        if not encoded:
            raise ValueError("missing audio")
        return base64.b64decode(encoded)
    except (ValueError, TypeError) as exc:
        raise HTTPException(502, f"火山方舟返回格式异常: {response.text[:500]}") from exc

async def minimax_tts(text: str, voice: str, style: str) -> bytes:
    key = os.getenv("MINIMAX_API_KEY")
    if not key:
        raise HTTPException(503, "未配置 MINIMAX_API_KEY")
    payload = {"model": os.getenv("MINIMAX_TTS_MODEL", "speech-2.8-hd"), "text": text,
               "voice_setting": {"voice_id": voice or "male-qn-qingse", "speed": 1, "vol": 5, "pitch": 0},
               "audio_setting": {"audio_sample_rate": 44100, "bitrate": 256000, "format": "mp3", "channel": 1}}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(f"{MINIMAX_BASE_URL}/v1/t2a_async_v2", headers=headers, json=payload)
        response.raise_for_status(); task_id = response.json().get("task_id")
        if not task_id: raise HTTPException(502, "MiniMax 未返回 task_id")
        import asyncio
        for _ in range(120):
            await asyncio.sleep(2)
            q = await client.get(f"{MINIMAX_BASE_URL}/v1/query/t2a_async_query_v2", params={"task_id": task_id}, headers=headers); q.raise_for_status(); data = q.json(); status = data.get("status", "").lower()
            if status == "success":
                f = await client.get(f"{MINIMAX_BASE_URL}/v1/files/retrieve", params={"file_id": data.get("file_id")}, headers=headers); f.raise_for_status(); url = f.json().get("file", {}).get("download_url")
                if not url: raise HTTPException(502, "MiniMax 未返回下载地址")
                a = await client.get(url); a.raise_for_status(); return a.content
            if status in {"failed", "expired"}: raise HTTPException(502, f"MiniMax 任务失败: {data}")
        raise HTTPException(504, "MiniMax TTS 任务超时")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

@app.get("/api/voices")
async def voices() -> dict[str, list[dict[str, str]]]:
    result = {"mimo": MIMO_VOICES, "volc": VOLC_VOICES, "minimax": []}
    source = ROOT / "Voice ID.md"
    if source.exists():
        pattern = re.compile(r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|")
        result["minimax"] = [{"id": m.group(2).strip(), "name": m.group(3).strip(), "language": m.group(1).strip()} for m in pattern.finditer(source.read_text(encoding="utf-8", errors="replace"))]
    return result


@app.post("/api/tts")
async def synthesize(provider: str = Form(...), text: str = Form(...), voice: str = Form(""), style: str = Form(""), sample: UploadFile | None = File(None)) -> dict[str, str]:
    text = text.strip()
    if not text:
        raise HTTPException(400, "请输入要合成的文本")
    if provider not in {"mimo", "volc", "minimax"}:
        raise HTTPException(400, "不支持的服务商")
    if sample is not None and sample.filename:
        suffix = Path(sample.filename).suffix.lower()
        if suffix not in {".wav", ".mp3"}:
            raise HTTPException(400, "音频样本仅支持 WAV 或 MP3")
        await sample.read()  # 第一版保留上传入口，火山订阅 TTS 仅使用文本合成
    parts = split_text(text)
    audio = bytearray()
    for part in parts:
        audio.extend(await (mimo_tts(part, voice or "mimo_default", style) if provider == "mimo" else volc_tts(part, voice, style) if provider == "volc" else minimax_tts(part, voice, style)))
    extension = "mp3" if provider == "minimax" else "wav"
    filename = f"{int(time.time())}_{provider}.{extension}"
    target = OUTPUT_DIR / filename
    target.write_bytes(audio)
    return {"url": f"/outputs/{filename}", "filename": filename, "parts": str(len(parts))}


@app.get("/outputs/{filename}")
async def output(filename: str) -> FileResponse:
    target = OUTPUT_DIR / clean_name(filename)
    if not target.exists() or target.parent != OUTPUT_DIR:
        raise HTTPException(404, "音频不存在")
    media_type = "audio/mpeg" if target.suffix == ".mp3" else "audio/wav"
    return FileResponse(target, media_type=media_type, filename=target.name)
