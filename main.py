"""
Note: Authorization should be handled via signed JWT tokens in the future. After
the validation of the token and authorization check, the same token can be used
to create the instances in Cordra. Until then, a usernam and password of an
Cordra account must be specified.
"""
# TODO: refactor this file into logically nicely separated Python files

# TODO: make more generic

from urllib.parse import urlparse
import requests
from jose import jwk, jwt, exceptions
import json

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form, Response, Header, BackgroundTasks
from typing import Optional
from starlette.responses import StreamingResponse
from pydantic import BaseModel, validator

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

# According to model training: https://github.com/2younis/plant-organ-detection/blob/master/train_net.py
CLASS_LIST = ['leaf', 'flower', 'fruit', 'seed', 'stem', 'root']

class ReturnType(str, Enum):
    json = 'json'
    image = 'image'
    cordra = 'cordra'

class Body(BaseModel):
    url: str
    linkedDigitalObjectId: Optional[str] = ''
    returnType: Optional[ReturnType] = ReturnType.json
    runAsync: Optional[bool] = False

    @validator('returnType')
    def must_have_specimen_id_and_async(cls, v, values):
        if v == 'cordra':
            if 'linkedDigitalObjectId' not in values or not values['linkedDigitalObjectId']:
                raise ValueError('A linkedDigitalObjectId must be provided with returnType cordra')
        return v

    @validator('runAsync')
    def must_only_be_used_with_cordra(cls, v, values):
        if v:
            if 'returnType' not in values or not values['returnType'] == 'cordra':
                raise ValueError('runAsync can only be used with returnType cordra')
        return v

class CordraClient:
    def __init__(self, base_url, verify_tls = True):
        if not base_url.endswith('/'):
            base_url += '/'
        self.base_url = base_url
        self.verify_tls = verify_tls

    def upload(self, data_dict, auth=None):
        request_url = self.base_url + 'objects?type=AnnotationList'
        if auth is not None:
            auth = requests.auth.HTTPBasicAuth(auth['username'], auth['password'])
        return requests.post(
                    request_url,
                    headers={'Content-Type': 'application/json'},
                    auth=auth,
                    data=json.dumps(data_dict),
                    verify=self.verify_tls)

def doc(docstring):
    def document(func):
        func.__doc__ = docstring
        return func
    return document

app = FastAPI()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml'))
cfg.merge_from_file('config/custom_model_config.yaml')

cfg.freeze()
MetadataCatalog.get('object_detection_service').set(thing_classes=CLASS_LIST)
predictor = DefaultPredictor(cfg)
with open('config/cordraConfig.json') as file:
    txt = file.read()
cordra_config = json.loads(txt)
cordra_credentials = cordra_config['credentials']
cordra_client = CordraClient(cordra_config['url'],cordra_config['verifyTls'])

"""
with open('config/keycloakConfig.json') as file:
    txt = file.read()
authConfig = json.loads(txt)
public_key = jwk.construct(authConfig["jwk"])


def authenticate(token):
    auth_success = False
    if isinstance(token, str) and token.startswith('Bearer '):
        try:
            decoded_token = jwt.decode(token, public_key, algorithms=[authConfig['jwk']['alg']], audience=authConfig['aud'][0])
        except exceptions.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail='Authentication failed')
        if 'iss' in decoded_token and decoded_token['iss'] == authConfig['iss']:
            auth_success = True
    if not auth_success:
        raise HTTPException(status_code=401, detail='Authentication failed')
"""

@app.get('/')
def read_root():
    return {'help': ('Submit your image as multipart/form-data with POST '
                    'operation to the /object-detection/image-upload endpoint '
                    'or submit a url (body as JSON) to the /classify_url endpoint')}

@app.post('/object-detection', summary="Run object detection on a provided image url")
@doc("""
    Upload a json object containing the URL to an image and optional metadata
    in order to run the object detection on that image. JSON body can have the
    following parameters:
    - **url**: (required) url to the image to process
    - **returnType**: (optional) json/image/cordra, default: json
    - **linkedDigitalObjectId**: (optional) the linkedDigitalObjectId that the generated
    annotations will be linked to, default: ''
    - **runAsync**: (optional) if true, will run the object detection
    asynchronously, default: false

    Note that when the returnType is cordra then a linkedDigitalObjectId must be
    provided and it is recommended to run with **runAsync** = true. The detected
    objects will be uploaded as AnnotationList objects to the Cordra instance %s
    """ % cordra_config['url'])
async def object_detection(
                body: Body,
                background_tasks: BackgroundTasks,
                authorization: Optional[str] = Header(None),):
    #authenticate(authorization)
    valid_url = urlparse(body.url)
    if not all([valid_url.scheme in ['file', 'http', 'https'], valid_url.netloc, valid_url.path]):
        raise HTTPException(status_code=400, detail='The provided value is not a valid URL')
    try:
        response = requests.get(body.url, timeout=10, verify=cordra_config['verifyTls'])
    except:
        raise HTTPException(status_code=400, detail='An error ocurred during fetching the image from the provided url')
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail='Unable to fetch image from the provided url')
    process_function = lambda: process_image(
                                response.content,
                                body.returnType,
                                linkedDigitalObjectId=body.linkedDigitalObjectId,
                                imageURI=body.url)
    if body.runAsync:
        background_tasks.add_task(process_function)
        return {
            'info': ('Started image detection, generated annotations will be '
                    'uploaded to ' + cordra_config['url'])}
    else:
        return process_function()

@app.post(
    '/object-detection/image-upload',
    summary="Run object detection on an uploaded image")
@doc("""
    Upload an image object as multipart formdata in order to run the object
    detection on it.

    Note that when the returnType is cordra then a linkedDigitalObjectId must be
    provided and it is recommended to run with **runAsync** = true. The detected
    objects will be uploaded as AnnotationList objects to the Cordra instance %s
    """ % cordra_config['url'])
async def object_detection_image_upload(
                payload: UploadFile = File(...),
                returnType: ReturnType = Form(ReturnType.json),
                linkedDigitalObjectId: Optional[str] = '',
                imageURI: Optional[str] = '',
                runAsync: Optional[bool] = False,
                authorization: Optional[str] = Header(None)):
    #authenticate(authorization)
    if runAsync:
        background_tasks.add_task(
                                process_image,
                                await payload.read(),
                                returnType,
                                linkedDigitalObjectId=linkedDigitalObjectId,
                                imageURI=imageURI)
        return {
            'info': ('Started image detection, generated annotations will be '
                    'uploaded to ' + cordra_config['url'])}
    else:
        return process_image(
                                await payload.read(),
                                returnType,
                                linkedDigitalObjectId=linkedDigitalObjectId,
                                imageURI=imageURI)

def process_image(img, returnType, linkedDigitalObjectId = '', imageURI = ''):
    img = convert_PIL_to_numpy(Image.open(io.BytesIO(img)), 'BGR')
    predictions = predictor(img)
    instances = predictions['instances']
    instances_result = []
    class_names = CLASS_LIST
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes
    scores = instances.scores.numpy()
    num_instances = len(boxes)
    for i in range(num_instances):
        instances_result.append({
            'class': class_names[classes[i]],
            'score': float(scores[i]),
            'boundingBox': [int(x) for x in boxes[i]]
        })
    if returnType == 'json':
        result = {'success': True}
        result['instances'] = instances_result
        return result
    elif returnType == 'image':
        img = img[:, :, ::-1]
        visualizer = Visualizer(img, MetadataCatalog.get('object_detection_service'),
                                instance_mode=ColorMode.SEGMENTATION)
        vis_output = visualizer.draw_instance_predictions(
            predictions=instances)
        img3 = Image.fromarray(vis_output.get_image())
        byte_io = io.BytesIO()
        img3.save(byte_io, format='jpeg')
        jpg_buffer = byte_io.getvalue()
        byte_io.close()
        return Response(jpg_buffer, media_type='image/jpeg')
    elif returnType == 'cordra':
        if not imageURI:
            raise HTTPException(status_code=400, detail='No imageURI provided')
        if not linkedDigitalObjectId:
            raise HTTPException(status_code=400, detail='No digital specimen ID provided')
        result_dict = {
                'detectedInstances': len(instances_result),
                'annotationListCreated': False
        }
        data_dict = {
                    'ods:linkedDigitalObject': linkedDigitalObjectId,
                    'ods:originatorType': 'machine-detected',
                    'ods:imageURI': imageURI,
                    'ods:annotations': instances_result
        }
        try:
            response = cordra_client.upload(data_dict, auth=cordra_credentials)
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=500, detail='Could not connect to Cordra server')
        if response.status_code in [200, 201]:
            result_dict['annotationListCreated'] = True
            result_dict['annotationList'] = cordra_config['url'] + '/objects/' + response.json()['id']
        else:
            result_dict['detail'] = response.json()
        return result_dict
