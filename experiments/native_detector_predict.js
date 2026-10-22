/* Core ML/Vision local-file inference. Callers keep private outputs in ignored data/. */
ObjC.import('Foundation');
ObjC.import('CoreML');
ObjC.import('Vision');

function run(argv) {
    if (argv.length !== 1) throw new Error('Expected local request JSON path');
    var error = Ref();
    var text = $.NSString.stringWithContentsOfFileEncodingError(argv[0], $.NSUTF8StringEncoding, error);
    var spec = JSON.parse(ObjC.unwrap(text));
    var configuration = $.MLModelConfiguration.alloc.init;
    configuration.computeUnits = 0;
    var model = $.MLModel.modelWithContentsOfURLConfigurationError($.NSURL.fileURLWithPath(spec.compiled_model), configuration, error);
    var visionModel = $.VNCoreMLModel.modelForMLModelError(model, error);
    var provider = $.MLDictionaryFeatureProvider.alloc.initWithDictionaryError(
        $({confidenceThreshold:0.01, iouThreshold:0.5}), error);
    visionModel.featureProvider = provider;
    var request = $.VNCoreMLRequest.alloc.initWithModel(visionModel);
    request.imageCropAndScaleOption = 2;
    var rows = [];
    spec.images.forEach(function(item) {
        var started = Date.now();
        var handler = $.VNImageRequestHandler.alloc.initWithURLOptions($.NSURL.fileURLWithPath(item.path), $({}));
        if (!handler.performRequestsError($.NSArray.arrayWithObject(request), error)) {
            throw new Error('Native image inference failed');
        }
        var results = request.results;
        var objects = [];
        for (var i = 0; i < Number(results.count); i++) {
            var observation = results.objectAtIndex(i);
            var labels = [];
            for (var j = 0; j < Number(observation.labels.count); j++) {
                var label = observation.labels.objectAtIndex(j);
                labels.push({label:ObjC.unwrap(label.identifier), confidence:Number(label.confidence)});
            }
            var box = observation.boundingBox;
            objects.push({labels:labels, box:[Number(box.origin.x), Number(box.origin.y), Number(box.size.width), Number(box.size.height)]});
        }
        rows.push({id:item.id, objects:objects, duration_ms:Date.now()-started});
    });
    return JSON.stringify({compute_units:Number(configuration.computeUnits), confidence_floor:0.01, iou_threshold:0.5, rows:rows});
}
