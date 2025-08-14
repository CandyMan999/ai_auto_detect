web: gunicorn app:app -k uvicorn.workers.UvicornWorker --workers 1 --threads 1 --timeout 120
worker: rq worker -u "$REDIS_TLS_URL?ssl_cert_reqs=none" facequeue nudityqueue
