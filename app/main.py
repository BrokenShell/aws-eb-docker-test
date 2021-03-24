from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import ocr
from app import routes


app = FastAPI(
    title='Test',
    description='Testing Docker with AWS EB',
    docs_url='/'
)

app.include_router(routes.router, tags=["Test Route"])
app.include_router(ocr.router, tags=['PDF Converter'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
