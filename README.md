# plant-detection-service

This is a demonstration of how to provide a trained machine learning model for plant organ detection as a micro service. This project is based on the work of @2younis and his repo [2younis/plant-organ-detection](https://github.com/2younis/plant-organ-detection). His work is described in the following publication:

> Younis et al. (2020): Detection and Annotation of Plant Organs from Digitized Herbarium Scans using Deep Learning
> https://arxiv.org/abs/2007.13106

The purpose of this repo is to create a micro service where a herbarium sheet image can be sent to via http, which then applies the trained model on the image and returns the detected plant organ instances in JSON format. Also an example is given of this service could interact with a [Cordra](https://www.cordra.org/index.html) instance.


## Setup
Requirement: Python3 with the venv module installed
```bash
# copy this repository
git clone https://github.com/jgrieb/plant-detection-service
# create a virtual environment
python3 -m venv plant-detection-service
cd plant-detection-service && source bin/activate
# install dependencies
pip install -r requirements.txt
# download the trained ML model
wget https://github.com/2younis/plant-organ-detection/releases/download/v1.0/model_final.pth -P data/
```


## Run
Run the service with `bin/uvicorn main:app` (you might want to add `--host 0.0.0.0 --port 8000` or another configuration)

You can then either access it by passing a url to an image:
`curl -XPOST localhost:8000/classify_url -H "Content-Type: application/json" -d '{"url":"http://oxalis.br.fgov.be/images/BR0/000/013/701/604/BR0000013701604.jpg"}'`

Or by uploading the image directly:
`curl -XPOST localhost:8000/classify_image -F 'payload=@images/BR0000013701604.jpg'`

#### Return type
By default JSON is returned, but you can change this so that the service returns the image with the detected bounding boxes drawn on top (in this case you should specify the output file with the `-o` flag)

`curl -XPOST localhost:8000/classify_url -H "Content-Type: application/json" -d '{"url":"http://oxalis.br.fgov.be/images/BR0/000/013/701/604/BR0000013701604.jpg", "returnType":"image"}' -o output.jpg`

`curl -XPOST localhost:8000/classify_image -F 'payload=@images/BR0000013701604.jpg' -F 'returnType=image' -o output.jpg`

Returned JSON data has the following format:
```json
{
  "success":true,
  "instances":[
    {
      "class":"leaf",
      "score":0.9999428987503052,
      "boundingBox":[2470,3957,3229,4348]
    }
  ]
}
```
where the boundingBox are pixel coordinates of [xmin, ymin, xmax, ymax].

## Interaction with Cordra
The microservice can be accessed by Cordra via HTTP requests. The idea is that a Digital Specimen in Cordra has an image attached as a payload and provides a [type method](https://www.cordra.org/documentation/design/type-methods.html) `createAnnotations`. When this type method is invoked on the object by an external client, Cordra sends a HTTP request to the plant detection service together with the image URL. The returned output from the microservice is processed by Cordra, serialized into RDF-jsonld and stored directly in the Digital Specimen object. The JavaScript for such a method would look like this:

```javascript
// Nashorn JDK cannot use XmlHttpRequest, therefore we must use Java Http requests
// see for reference: https://gist.github.com/billybong/a462152889b6616deb02

var cordra = require("cordra");
var cordraBaseUri =  cordra.get('design').content.handleMintingConfig.baseUri;
// set the URI to the plant detection uri, in this example running on the same host as Cordra
var imageServiceUri = 'http://127.0.0.1:8000';

/* TODO: also needed to provide the correct context with the Cordra object when annotations exist, e.g.:
"@context": [
    "http://iiif.io/api/presentation/3/context.json",
    {"ods": "http://github.com/hardistyar/openDS/ods-ontology/terms/"},
    {"dwc": "http://rs.tdwg.org/dwc/terms/"},
]
*/
exports.methods = {};
exports.methods.createAnnotations = createAnnotations;

function createAnnotations(object, context){
    if (!context.params || typeof(context.params) !== 'object' || !('payload' in context.params)){
        throw new cordra.CordraError({'message': 'Missing parameter 'payload''}, 400);
    }
    var con = new java.net.URL(imageServiceUri + '/classify_url').openConnection();
    con.setConnectTimeout(30000); // 30 seconds
    con.setReadTimeout(30000); // 30 seconds
    var payloadUrl = cordraBaseUri + '/objects/' + object.id + '?payload=' + context.params.payload;
    var data = JSON.stringify({'url': payloadUrl});
    var response = {};
    console.log('Sending request to micro service');
    try {
        write(con, data);
        response = JSON.parse(read(con.inputStream));
    } catch (e){
        throw 'An error occured during processing ' + e;
    }
    if(con.responseCode === 200 && response.success && Array.isArray(response.instances)){

        var graph = object.content['@graph'];
        var existingAnnotationPages = graph.filter(function(i){
            return i['@type'] === 'AnnotationPage'
            || i['type'] === 'AnnotationPage';
        });
        var counter = 0;
        if (existingAnnotationPages.length > 0) {
            counter = existingAnnotationPages[0].items.length; // there should be only one
        }
        var items = [];
        for (var i=0; i<response.instances.length; i++){
            // boundingBox is returned as [xmin, ymin, xmax, ymax]
            var bb = response.instances[i].boundingBox;
            var x = bb[0]; // xmin
            var y = bb[1]; // ymin
            var w = bb[2] - x;
            var h = bb[3] - y;
            var item = {
                '@id': cordraBaseUri + '/objects/' + object.id + '/annotations/' + counter,
                '@type': 'Annotation',
                'oa:hasSelector': {
                    'type': 'FragmentSelector',
                    'value': 'xywh='+[x,y,w,h].join(',')
                },
                'dwc:measurementType': 'automated plant organ classification',
                'dwc:measurementValue': response.instances[i].class,
                'dwc:measurementAccuracy': response.instances[i].score
            }
            items.push(item);
            counter += 1;
        }

        if (existingAnnotationPages.length > 0){
            existingAnnotationPages[0].items.push(items);
        } else {
            object.content['@graph'].push({
                '@id': cordraBaseUri + '/objects/' + object.id + '/annotations/',
                '@type': 'AnnotationPage',
                'items': items
            });
        }
        return {
          'success': true,
          'message': 'Detected and created ' + items.length + ' new annotations'
        };
    } else {
       throw new cordra.CordraError({'message': response.detail || 'An error occurred during image classification'}, 409);
    }
}

function read(inputStream){
    var inReader = new java.io.BufferedReader(new java.io.InputStreamReader(inputStream));
    var inputLine;
    var response = new java.lang.StringBuffer();
    while ((inputLine = inReader.readLine()) !== null) {
           response.append(inputLine);
    }
    inReader.close();
    return response.toString();
}

function write(con, data){
    con.requestMethod = 'POST';
    con.setRequestProperty( 'Content-Type', 'application/json; charset=utf-8');
    con.doOutput=true;
    var wr = new java.io.DataOutputStream(con.outputStream);
    wr.writeBytes(data);
    wr.flush();
    wr.close();
}
```


## Discussion
This example above stores the detected plant organ instances and their bounding boxes as `http://www.w3.org/ns/oa:annotation`(respective the collection of them as one `AnnotationPage`), according to the [IIIF Presentation API 3.0](https://iiif.io/api/presentation/3.0/#55-annotation-page). It still needs to be evaluated whether this is the correct vocabulary.

Also it needs to be discusses whether Cordra should be responsible for the RDF serialization (as above) or whether the plant detection microservice should return the results directly in RDF.
