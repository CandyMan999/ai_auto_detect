
web: gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 60
worker: rq worker -u "$REDIS_TLS_URL?ssl_cert_reqs=none" facequeue
