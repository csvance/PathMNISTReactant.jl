# Export: what a client sends and what comes back. The model converts UInt8 pixels inside
# its own program, so no `export_preprocess` is needed, and only the logits ship.
#region export
ReactantNitro.export_inputs(::PathMNIST) = [ExportSpec("img", UInt8, [224, 224, 3, 1])]
ReactantNitro.export_outputs(::PathMNIST) = [ExportSpec("logits")]
#endregion
