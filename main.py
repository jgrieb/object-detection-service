from urllib.parse import urlparse
from requests import get

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form, Response
from typing import Optional
from starlette.responses import StreamingResponse
from pydantic import BaseModel

from PIL import Image
import io
from numpy import asarray

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2 import model_zoo
from enum import Enum

class ReturnType(str, Enum):
    json = 'json'
    image = 'image'

class Body(BaseModel):
    url: str
    returnType: Optional[ReturnType] = ReturnType.json

# According to model training: https://github.com/2younis/plant-organ-detection/blob/master/train_net.py
ORGAN_LIST = ['leaf', 'flower', 'fruit', 'seed', 'stem', 'root']
THING_COLORS = [(0, 255, 0), (0, 0, 255), (255, 153, 0),
                       (255, 255, 0), (128, 64, 0), (255, 0, 0)]

app = FastAPI()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml'))
cfg.merge_from_list(['MODEL.WEIGHTS', 'data/model_final.pth', 'MODEL.DEVICE', 'cpu'])

# Specific config of trained model https://github.com/2younis/plant-organ-detection/blob/master/configs/config.py
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.RPN.NMS_THRESH = 0.6
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.25
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512, 1024]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

# Set score_threshold for builtin models
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

# Set correct thing classes
cfg.DATASETS.TEST = ('plant_detection',)
cfg.freeze()
MetadataCatalog.get('plant_detection').set(thing_classes=ORGAN_LIST,
                                            thing_colors=THING_COLORS)
predictor = DefaultPredictor(cfg)

@app.get('/')
def read_root():
    return {'help': 'Submit your image as multipart/form-data with POST '\
    + 'operation to the /classify_image endpoint or submit a url (body as JSON) to the /classify_url endpoint'}

@app.post('/classify_url')
def read_item(body: Body):
    valid_url = urlparse(body.url)
    if not all([valid_url.scheme in ['file', 'http', 'https'], valid_url.netloc, valid_url.path]):
        raise HTTPException(status_code=400, detail='The provided value is not a valid URL')
    try:
        response = get(body.url, timeout=10)
    except:
        raise HTTPException(status_code=400, detail='An error ocurred during fetching the image from the provided url')
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail='Unable to fetch image from the provided url')
    return process_image(response.content, body.returnType)

@app.post('/classify_image')
async def read_item(payload: UploadFile = File(...), returnType: ReturnType = Form(ReturnType.json)):
    return process_image(await payload.read(), returnType)

def process_image(img, returnType):
    img = convert_PIL_to_numpy(Image.open(io.BytesIO(img)), 'BGR')
    predictions = predictor(img)
    instances = predictions['instances']
    if returnType == 'json':
        class_names = ORGAN_LIST
        result = {'success': False}
        if len(instances) > 0:
            result['success'] = True
            boxes = instances.pred_boxes.tensor.numpy()
            classes = instances.pred_classes
            num_instances = len(boxes)
            scores = instances.scores.numpy()
            instances_result = []
            for i in range(num_instances):
                instances_result.append({
                    'class': class_names[classes[i]],
                    'score': float(scores[i]),
                    'boundingBox': [int(x) for x in boxes[i]]
                })
            result['instances'] = instances_result
        return result
    elif returnType == 'image':
        img = img[:, :, ::-1]
        visualizer = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TEST[0]),
                                instance_mode=ColorMode.SEGMENTATION)
        vis_output = visualizer.draw_instance_predictions(
            predictions=instances)
        img3 = Image.fromarray(vis_output.get_image())
        byte_io = io.BytesIO()
        img3.save(byte_io, format='jpeg')
        jpg_buffer = byte_io.getvalue()
        byte_io.close()
        return Response(jpg_buffer, media_type='image/jpeg')
