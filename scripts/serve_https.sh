#!/bin/bash
# Serve the app over HTTPS so voice input works on phones and other computers.
# Browsers only allow the microphone on https:// pages (or localhost).
# Uses a self-signed certificate: the first visit shows a warning - tap
# "Show details" -> "visit this website" (Safari) or "Advanced" -> "Proceed" (Chrome).
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-8443}"
mkdir -p .certs
IPS=$(ifconfig | awk '/inet / && $2 != "127.0.0.1" {print $2}')
SAN="DNS:localhost,IP:127.0.0.1"
for ip in $IPS; do SAN="$SAN,IP:$ip"; done
# regenerate when the network address changes
if [ ! -f .certs/cert.pem ] || [ "$(cat .certs/san 2>/dev/null)" != "$SAN" ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=VEX Scout AI" \
    -keyout .certs/key.pem -out .certs/cert.pem -addext "subjectAltName=$SAN" 2>/dev/null
  echo "$SAN" > .certs/san
fi
echo "VEX Scout AI over HTTPS:"
for ip in $IPS; do echo "  https://$ip:$PORT"; done
echo "  https://localhost:$PORT"
exec .venv/bin/streamlit run app.py --server.headless true --server.port "$PORT" \
  --server.sslCertFile .certs/cert.pem --server.sslKeyFile .certs/key.pem
