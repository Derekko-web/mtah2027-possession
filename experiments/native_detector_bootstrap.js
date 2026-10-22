/* Native Core ML metadata only. No image or competition input is read. */
ObjC.import('Foundation');
ObjC.import('CoreML');
ObjC.import('Vision');

function run(argv) {
    if (argv.length !== 2) throw new Error('Expected model and fresh compiled-output paths');
    var modelURL = $.NSURL.fileURLWithPath(argv[0]);
    var outputURL = $.NSURL.fileURLWithPath(argv[1]);
    var error = Ref();
    var compiled = $.MLModel.compileModelAtURLError(modelURL, error);
    if (!ObjC.unwrap(compiled.path)) throw new Error('Native model compilation failed');
    var copied = $.NSFileManager.defaultManager.copyItemAtURLToURLError(compiled, outputURL, error);
    if (!copied) throw new Error('Compiled-model preservation failed');
    var configuration = $.MLModelConfiguration.alloc.init;
    configuration.computeUnits = 0;
    var model = $.MLModel.modelWithContentsOfURLConfigurationError(outputURL, configuration, error);
    var description = model.modelDescription;
    var inputs = ObjC.deepUnwrap(description.inputDescriptionsByName.allKeys);
    var outputs = ObjC.deepUnwrap(description.outputDescriptionsByName.allKeys);
    var metadata = ObjC.deepUnwrap(description.metadata);
    var summary = {};
    inputs.forEach(function(name) {
        summary[name] = ObjC.unwrap(description.inputDescriptionsByName.objectForKey(name).description);
    });
    outputs.forEach(function(name) {
        summary[name] = ObjC.unwrap(description.outputDescriptionsByName.objectForKey(name).description);
    });
    return JSON.stringify({compiled_model:argv[1], compute_units:configuration.computeUnits,
        input_names:inputs, output_names:outputs, metadata:metadata, features:summary});
}
