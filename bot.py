# bot.py
import os
import json
import requests

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
NVIDIA_API = "https://integrate.api.nvidia.com/v1/chat/completions"

OFFSET_FILE = "offset.json"
CHUNK_SIZE = 2000
MAX_CHUNKS = 5

ROUTER_MODEL = "meta/llama-3.1-8b-instruct"
SPECIALIST_MODEL = "meta/llama-3.1-70b-instruct"

ROUTER_SYS = (
    "You are a strict classifier. Read the log excerpt and output EXACTLY ONE WORD: "
    "INFRASTRUCTURE, APPLICATION, or DATABASE. No punctuation. No explanation."
)

ANALYST_SYS_TEMPLATE = (
    "You are a terse {cat} systems analyst. Read the log chunk. "
    "Output only new terse bullet fragments (max 3, under 12 words each) noting root-cause clues. "
    "No intros, no summaries, no repeated points, no markdown."
)

SYNTH_SYS = (
    "You are a senior on-call engineer. Using the analyst notes below, output STRICT plain text "
    "in exactly this format and nothing else, no preamble:\n"
    "Status: <one line>\n"
    "Core Issue: <max 2 sentences>\n"
    "Git Patch/Fix:\n"
    "```\n<minimal diff or code fix>\n```"
)


def load_offset():
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE) as f:
            return json.load(f).get("offset", 0)
    return 0


def save_offset(offset):
    with open(OFFSET_FILE, "w") as f:
        json.dump({"offset": offset}, f)


def get_updates(offset):
    r = requests.get(
        f"{TELEGRAM_API}/getUpdates",
        params={"offset": offset, "timeout": 0},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("result", [])


def send_message(chat_id, text):
    for i in range(0, len(text), 4000):
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": text[i : i + 4000],
                "parse_mode": "Markdown",
            },
            timeout=15,
        )


def get_log_text(message):
    if "document" in message:
        file_id = message["document"]["file_id"]
        r = requests.get(
            f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=15
        )
        path = r.json()["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}"
        return requests.get(file_url, timeout=15).text
    return message.get("text", "")


def chunk_log(text, size=CHUNK_SIZE):
    text = text.strip()
    while text:
        if len(text) <= size:
            yield text
            return
        split_at = text.rfind("\n", 0, size)
        if split_at <= 0:
            split_at = size
        yield text[:split_at]
        text = text[split_at:].lstrip("\n")


def call_nvidia(model, system, user, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(NVIDIA_API, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def route_category(first_chunk):
    out = call_nvidia(ROUTER_MODEL, ROUTER_SYS, first_chunk[:CHUNK_SIZE], max_tokens=5).upper()
    for cat in ("INFRASTRUCTURE", "APPLICATION", "DATABASE"):
        if cat in out:
            return cat
    return "APPLICATION"


def diagnose(log_text):
    chunks = list(chunk_log(log_text))
    if not chunks:
        return (
            "Status: No log content received.\n"
            "Core Issue: Empty input.\n"
            "Git Patch/Fix:\n```\nN/A\n```"
        )

    category = route_category(chunks[0])
    analyst_sys = ANALYST_SYS_TEMPLATE.format(cat=category.lower())

    notes = []
    for chunk in chunks[:MAX_CHUNKS]:
        notes.append(call_nvidia(SPECIALIST_MODEL, analyst_sys, chunk, max_tokens=120))

    combined_notes = "\n".join(notes)
    return call_nvidia(
        SPECIALIST_MODEL,
        SYNTH_SYS,
        f"Category: {category}\nAnalyst notes:\n{combined_notes}",
        max_tokens=400,
    )


def main():
    offset = load_offset()
    updates = get_updates(offset)
    if not updates:
        return

    new_offset = offset
    for upd in updates:
        new_offset = upd["update_id"] + 1
        message = upd.get("message")
        if not message:
            continue

        chat_id = message["chat"]["id"]
        ALLOWED_USER_ID = int(os.environ.get("TELEGRAM_USER_ID", 0))
        if message["from"]["id"] != ALLOWED_USER_ID:
            continue
        text = message.get("text", "")

        if text.strip().lower() in ("/start", "/help"):
            send_message(chat_id, "Send a log file or paste error logs to diagnose.")
            continue

        log_text = get_log_text(message)
        if not log_text.strip():
            continue

        try:
            result = diagnose(log_text)
        except Exception as e:
            result = f"Status: Pipeline error.\nCore Issue: {e}\nGit Patch/Fix:\n```\nN/A\n```"

        send_message(chat_id, result)

    save_offset(new_offset)


if __name__ == "__main__":
    main()