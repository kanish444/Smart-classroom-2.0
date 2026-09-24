import os
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from loguru import logger

from app.api import router as api_router
from app.schemas import ApiResponse
from config.settings import get_settings


def create_app() -> FastAPI:
    """Factory creating and configuring the SmartClass Vision AI FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="SmartClass Vision AI - Smart Board Dashboard & Export APIs",
        description="High-Accuracy Single-Camera Multi-Student Face Identification & Attendance System",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 1. CORS Middleware
    origins = [o.strip() for o in settings.dashboard.cors_origins.split(",") if o.strip()]
    if not origins:
        origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Global Exception Handlers conforming to ApiResponse schema
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code_str = f"HTTP_{exc.status_code}"
        return JSONResponse(
            status_code=exc.status_code,
            content=ApiResponse.fail(code=code_str, message=str(exc.detail)).model_dump()
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        first_error = exc.errors()[0] if exc.errors() else {"msg": "Validation error"}
        msg = f"{first_error.get('loc', ['field'])[-1]}: {first_error.get('msg', 'invalid')}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ApiResponse.fail(code="VALIDATION_ERROR", message=msg).model_dump()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse.fail(code="INTERNAL_SERVER_ERROR", message="An internal server error occurred.").model_dump()
        )

    # 3. Include API router
    app.include_router(api_router)

    # 4. Static Files & Root HTML Page
    dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
    static_dir = os.path.join(dashboard_dir, "static")
    os.makedirs(static_dir, exist_ok=True)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        index_file = os.path.join(dashboard_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
    @app.on_event("startup")
    def on_startup():
        logger.info("SmartClass Vision AI starting up - initializing camera pipeline...")
        from app.state import get_app_state
        state = get_app_state()
        state.start_camera_worker()

    @app.on_event("shutdown")
    def on_shutdown():
        logger.info("SmartClass Vision AI shutting down - releasing camera resources...")
        from app.state import get_app_state
        state = get_app_state()
        state.stop_camera_worker()

    return app


# Application entrypoint for ASGI runners (uvicorn app.main:app)
app = create_app()
