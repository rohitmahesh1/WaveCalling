(* Start clean *)
ClearAll["Global`*"];

(* -------- Args & defaults -------- *)
args      = Rest @ $ScriptCommandLine;
modelsDir = If[Length[args] >= 1, args[[1]], "models"];
outDir    = If[Length[args] >= 2, args[[2]], "export"];
inH       = If[Length[args] >= 3, ToExpression@args[[3]], 256];
inW       = If[Length[args] >= 4, ToExpression@args[[4]], 512];
inColor   = If[Length[args] >= 5, args[[5]], "Grayscale"];  (* "Grayscale" or "RGB" *)

(* -------- Validation -------- *)
If[!DirectoryQ[modelsDir],
  Print["ERROR: Models directory not found: ", modelsDir];
  Exit[1];
];

CreateDirectory[outDir, CreateIntermediateDirectories -> True];

(* -------- Mapping (src -> target filename) -------- *)
names = <|
  "Unidrectional_Segmentation_Module_V1_0" -> "uni_seg.onnx",
  "Bidirectional_Segmentation_Module_V1_1" -> "bi_seg.onnx",
  "Classification_Module_V1_0"             -> "classifier.onnx",
  "Decision_Module_V1_0"                   -> "decision.onnx"
|>;

(* -------- Helpers -------- *)
fixInputSize[net_, size : {h_Integer, w_Integer}, color_String] := Module[
  {ports, key, enc},
  ports = Quiet @ NetInformation[net, "InputPorts"];
  key   = If[AssociationQ@ports && KeyExistsQ[ports, "Input"],
             "Input",
             If[AssociationQ@ports && Length@ports >= 1, First@Keys@ports, None]
          ];
  If[key === None, Return[net]];
  enc = NetEncoder[{"Image", size, color}];
  NetReplacePart[net, key -> enc]
];

exportOne[src_, tgt_] := Module[{srcPath, tgtPath, net, ok},
  srcPath = FileNameJoin[{modelsDir, src}];
  tgtPath = FileNameJoin[{outDir,    tgt}];

  If[!FileExistsQ[srcPath],
    Print["WARN: Missing model: ", srcPath];
    Return[$Failed];
  ];

  Print["Exporting: ", src, " -> ", tgt];
  net = Check[
    Import[srcPath],
    Print["ERROR: Failed to import ", srcPath]; Return[$Failed];
  ];

  net = fixInputSize[net, {inH, inW}, inColor];

  ok = Check[
    Export[tgtPath, net, "ONNX"],
    Print["ERROR: Failed to export to ", tgtPath]; Return[$Failed];
  ];

  ok
];

(* -------- Run -------- *)
KeyValueMap[exportOne, names];

Print["Done. Saved to ", outDir];
Exit[0];