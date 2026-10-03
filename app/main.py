import os
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from loguru import logger

from contextlib import asynccontextmanager
from app.api import router as api_router
from app.schemas import ApiResponse
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Guaranteed lifecycle manager ensuring background camera worker cleanly starts and releases."""
    logger.info("SmartClass Vision AI starting up - initializing camera pipeline...")
    from app.state import get_app_state
    state = get_app_state()
    state.start_camera_worker()
    try:
        yield
    finally:
        logger.info("SmartClass Vision AI shutting down - releasing camera resources...")
        state.stop_camera_worker()


def create_app() -> FastAPI:
    """Factory creating and configuring the SmartClass Vision AI FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="SmartClass Vision AI - Smart Board Dashboard & Export APIs",
        description="High-Accuracy Single-Camera Multi-Student Face Identification & Attendance System",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # 1. CORS Middleware
    origins = [o.strip().rstrip("/") for o in settings.dashboard.cors_origins.split(",") if o.strip()]
    if not origins:
        origins = ["*"]

    if settings.environment in ("production", "prod") and origins == ["*"]:
        logger.warning(
            "CORS is configured with wildcard '*' and allow_credentials=True in production. "
            "For production HTTPS, specify explicit origins via DASHBOARD_CORS_ORIGINS."
        )

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

    # 3. Include API routers
    app.include_router(api_router)
    from app.api_rbac import router as rbac_router
    app.include_router(rbac_router)

    # 4. Static Files & Root HTML Pages
    dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
    static_dir = os.path.join(dashboard_dir, "static")
    os.makedirs(static_dir, exist_ok=True)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/service-worker.js", include_in_schema=False)
    async def serve_service_worker():
        """Serves root PWA service worker with root scope permissions."""
        sw_file = os.path.join(dashboard_dir, "service-worker.js")
        if os.path.exists(sw_file):
            return FileResponse(
                sw_file,
                media_type="application/javascript",
                headers={"Service-Worker-Allowed": "/"}
            )
        raise HTTPException(status_code=404, detail="Service worker not found")

    @app.get("/", include_in_schema=False)
    @app.get("/login", include_in_schema=False)
    async def serve_login(request: Request):
        """Authoritative flow: Root URL serves LOGIN PAGE directly. Authenticated sessions route to role dashboard."""
        token = request.cookies.get("access_token")
        if token:
            from app.core.security import decode_access_token
            payload = decode_access_token(token)
            if payload:
                role = payload.get("role")
                if role == "hod":
                    return RedirectResponse(url="/hod/dashboard", status_code=303)
                elif role == "class_advisor":
                    return RedirectResponse(url="/advisor/dashboard", status_code=303)
        return FileResponse(os.path.join(dashboard_dir, "login.html"))

    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard(request: Request):
        """Dashboard redirect based on active authenticated role or back to login."""
        token = request.cookies.get("access_token")
        if token:
            from app.core.security import decode_access_token
            payload = decode_access_token(token)
            if payload:
                role = payload.get("role")
                if role == "hod":
                    return RedirectResponse(url="/hod/dashboard", status_code=303)
                elif role == "class_advisor":
                    return RedirectResponse(url="/advisor/dashboard", status_code=303)
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/hod/dashboard", include_in_schema=False)
    async def serve_hod_dashboard():
        hod_file = os.path.join(dashboard_dir, "hod_dashboard.html")
        if os.path.exists(hod_file):
            return FileResponse(hod_file)
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/advisor/dashboard", include_in_schema=False)
    async def serve_advisor_dashboard():
        advisor_file = os.path.join(dashboard_dir, "advisor_dashboard.html")
        if os.path.exists(advisor_file):
            return FileResponse(advisor_file)
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/smartboard", include_in_schema=False)
    @app.get("/live", include_in_schema=False)
    @app.get("/classroom/monitor", include_in_schema=False)
    async def serve_smartboard_monitor():
        index_file = os.path.join(dashboard_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return RedirectResponse(url="/login", status_code=303)

    return app


# Application entrypoint for ASGI runners (uvicorn app.main:app)
app = create_app()
