import modal
import subprocess
import os

image = modal.Image.from_registry(
    "gradientservice/parallax:latest",
    add_python="3.11",
)

app = modal.App("deep-scholar-parallax")
model_volume = modal.Volume.from_name("parallax-models", create_if_missing=True)
CACHE_PATH = "/vol/model_cache"

MINUTES = 60
PORT = 3001

# Use public model - no HF token needed
# 7B offers much better reasoning than 0.5B
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_PATH: model_volume},
    env={
        "HF_HOME": CACHE_PATH,
        "PYTHONUNBUFFERED": "1",
    },
    timeout=10 * MINUTES,  # how long to wait for cold start
    scaledown_window=15 * MINUTES,  # how long to stay up with no traffic
    max_containers=1,
    min_containers=1,
)
@modal.concurrent(max_inputs=32)  # how many requests one replica can handle
@modal.web_server(port=PORT, startup_timeout=10 * MINUTES)
def run_parallax():
    print(f"🚀 Starting Parallax Engine Core with {MODEL_NAME} on port {PORT}")

    cmd = [
        "/usr/bin/python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--mem-fraction-static",
        "0.8",
    ]

    print("Starting server process:", " ".join(cmd))
    # IMPORTANT: fire-and-forget, do NOT block here
    subprocess.Popen(cmd)

    # Let Modal know startup is done; web_server will now route HTTP traffic
    return "OK"
