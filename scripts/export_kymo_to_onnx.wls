(* Usage: wolframscript -file export_kymo_to_onnx.wls models export *)
modelsDir = If[Length[$ScriptCommandLine] > 1, $ScriptCommandLine[[2]], "models"];
outDir    = If[Length[$ScriptCommandLine] > 2, $ScriptCommandLine[[3]], "export"];
CreateDirectory[outDir, CreateIntermediateDirectories -> True];

names = {
  "Unidrectional_Segmentation_Module_V1_0" -> "uni_seg.onnx",
  "Bidirectional_Segmentation_Module_V1_1" -> "bi_seg.onnx",
  "Classification_Module_V1_0"             -> "classifier.onnx",
  "Decision_Module_V1_0"                   -> "decision.onnx"
};

fixInputSize[net_] := Module[{ports, enc},
  ports = Quiet@NetInformation[net, "InputPorts"];
  If[KeyExistsQ[ports, "Input"],
    (* You can change {256,512} later; just pick a multiple of 16. *)
    enc = NetEncoder[{"Image", {256, 512}, "Grayscale"}];
    net = NetReplacePart[net, "Input" -> enc];
  ];
  net
];

Do[
  Print["Exporting: ", src, " -> ", tgt];
  net = Import[FileNameJoin[{modelsDir, src}]];
  net = fixInputSize[net];
  Export[FileNameJoin[{outDir, tgt}], net]; (* ONNX by extension *)
, {src -> tgt, names}];

Print["Done. Saved to ", outDir];
