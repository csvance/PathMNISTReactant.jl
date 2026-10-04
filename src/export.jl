# Export: what a client sends and what comes back. The model already converts UInt8 pixels to
# normalized Float32 inside its own program, so the wire tensor is the batch field `forward`
# declares and no `export_preprocess` is needed. Of `forward`'s outputs only the logits ship;
# the kinetic energy and motion stay in training.
ReactantNitro.export_inputs(::PathMNIST) = [ExportSpec("img", UInt8, [224, 224, 3, 1])]
ReactantNitro.export_outputs(::PathMNIST) = [ExportSpec("logits")]
