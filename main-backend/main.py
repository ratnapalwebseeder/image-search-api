from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from brotli_asgi import BrotliMiddleware  # https://github.com/fullonic/brotli-asgi
# from fastapi.staticfiles import StaticFiles # static html files deploying

app = FastAPI(openapi_url="/api/openapi.json")

# allow cors - from https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Filename", "x-filename"]
)

# enable Brotli compression. Better for json payloads, supported by most browsers. Fallback to gzip by default. from https://github.com/fullonic/brotli-asgi
app.add_middleware(BrotliMiddleware)

# for Authorization: Bearer token header
# security = HTTPBearer()

# can add modules having api calls below
import search_images