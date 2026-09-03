# Run Whisper Speech API on Ubuntu

Open a terminal and follow these steps in order.

## Step 1: Navigate to the project directory

```bash
cd Whisper-Speech-Api
```

## Step 2: Activate the Python virtual environment

```bash
source .venv/bin/activate
```

After successful activation, `(.venv)` will appear at the beginning of the terminal prompt.

## Step 3: Start the HTTPS server

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

To stop the server, press `Ctrl+C`.

To exit the Python virtual environment, run:

```bash
deactivate
```

> **Note:** If port `443` is already in use, stop the processes using ports `80` and `443`, then run Step 3 again:

```bash
sudo fuser -k 80/tcp 443/tcp
```

## Monitor GPU usage every 0.5 seconds

Open another terminal and run:

```bash
watch -n 0.5 nvidia-smi
```

Press `Ctrl+C` to stop monitoring.
