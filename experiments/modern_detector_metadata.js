/* Public model metadata only; no competition input. */
ObjC.import('Foundation');
ObjC.import('CoreML');
function run(argv) {
    var error = Ref();
    var configuration = $.MLModelConfiguration.alloc.init;
    configuration.computeUnits = 0;
    var model = $.MLModel.modelWithContentsOfURLConfigurationError(
        $.NSURL.fileURLWithPath(argv[0]), configuration, error);
    var description = model.modelDescription;
    var constraint = description.inputDescriptionsByName.objectForKey('image').imageConstraint;
    return JSON.stringify({width:Number(constraint.pixelsWide), height:Number(constraint.pixelsHigh),
        outputs:ObjC.deepUnwrap(description.outputDescriptionsByName.allKeys)});
}
