from fastapi import FastAPI, File, Request, UploadFile
from starlette.responses import Response
from fastapi.responses import HTMLResponse
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List

from .segmentation import EyeDiseaseSegmentation

app = FastAPI(
    title = "Eye disease Segmentation using deep learning",
    description = "System which helps doctor to give more correct decisions by segment all kind of lesion in fundus image",
    version = "0.1"
)

def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap

# @app.get("/")
# @construct_response
# def _index(request: Request):
#     """Health check."""
#     response = {
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#         "data": {},
#     }
#     return response

@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
    content = """
    <body>
    <form action="/files/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    </body>
        """
    return HTMLResponse(content=content)

# @app.on_event("startup")
# def load_artifacts():
#     global artifacts
#     artifacts = main.load_artifacts(model_dir=config.MODEL_DIR)
#     logger.info("Ready for inference!")
    
# @app.post("/segmentation/{lesion_type}")
# def get_segmentation_map(file: bytes = File(...)):
#     '''Get segmentation maps from image file'''
#     segmented_image = get_segments(model, file)
#     bytes_io = io.BytesIO()
#     segmented_image.save(bytes_io, format='PNG')
#     return Response(bytes_io.getvalue(), media_type="image/png")
