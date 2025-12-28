from fastapi import FastAPI, HTTPException, Request
from app.core.events import lifespan
from app.routers import api_router
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Convergent", lifespan=lifespan)
# Allow all origins (any domain or IP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- this allows any domain
    allow_credentials=True,  # allow cookies/auth headers
    allow_methods=["*"],  # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # allow all headers
)
app.include_router(api_router)


@app.get("/test")
async def root():
    return {"message": "Hello World"}


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status_code": exc.status_code, "message": exc.detail},
    )
