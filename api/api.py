from fastapi import FastAPI, File
import uvicorn
from starlette.responses import Response
import io
from .segmentation import *


model = get_model()

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id):

    return {"item_id": item_id}

@app.post("/segmentation")
def get_segmentation_map(file: bytes = File(...)):
    '''Get segmentation maps from image file'''
    segmented_image = get_segments(model, file)
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io, format='PNG')
    return Response(bytes_io.getvalue(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run('api:app', host="0.0.0.0", port=8000, reload=True)  