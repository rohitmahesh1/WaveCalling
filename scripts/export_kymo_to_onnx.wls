(* scripts/export_kymo_to_onnx.wls *)

ClearAll["Global`*"];

(* Args *)
args      = Rest @ $ScriptCommandLine;
modelsDir = If[Length[args] >= 1, args[[1]], "models"];
outDir    = If[Length[args] >= 2, args[[2]], "export"];
inH       = If[Length[args] >= 3, ToExpression@args[[3]], 256];  (* for seg nets *)
inW       = If[Length[args] >= 4, ToExpression@args[[4]], 512];

If[!DirectoryQ[modelsDir], Print["ERROR: Models directory not found: ", modelsDir]; Exit[1]];
CreateDirectory[outDir, CreateIntermediateDirectories -> True];

(* Per-model export targets + input tensor dims {C,H,W} *)
specs = <|
  "Unidrectional_Segmentation_Module_V1_0" -> <|"onnx"->"uni_seg.onnx", "dims"->{1, inH, inW}|>,
  "Bidirectional_Segmentation_Module_V1_1" -> <|"onnx"->"bi_seg.onnx",  "dims"->{1, inH, inW}|>,
  "Classification_Module_V1_0"             -> <|"onnx"->"classifier.onnx","dims"->{1, 64, 64}|>,
  "Decision_Module_V1_0"                   -> <|"onnx"->"decision.onnx",  "dims"->{3, 48, 48}|>
|>;

(* Replace the input encoder with a Tensor encoder that exactly matches the net’s expected array *)
fixInputToTensor[net_, dims : {c_, h_, w_}] := Module[{ports, key},
  ports = Quiet @ NetInformation[net, "InputPorts"];
  key   = If[AssociationQ@ports && KeyExistsQ[ports, "Input"],
             "Input",
             If[AssociationQ@ports && Length@ports >= 1, First@Keys@ports, None]];
  If[key === None, Return[net]];
  NetReplacePart[net, key -> NetEncoder[{"Tensor", dims, "Real"}]]
];

exportOne[name_, conf_] := Module[{srcPath, tgtPath, net, ok},
  srcPath = FileNameJoin[{modelsDir, name}];
  tgtPath = FileNameJoin[{outDir, conf["onnx"]}];

  If[!FileExistsQ[srcPath],
    Print["WARN: Missing model: ", srcPath]; Return[$Failed];
  ];

  Print["Exporting: ", name, "  →  ", conf["onnx"], "  with dims ", conf["dims"]];
  net = Check[ Import[srcPath], Print["ERROR: Failed to import ", srcPath]; Return[$Failed]; ];

  (* set an explicit fixed shape for ONNX *)
  net = fixInputToTensor[net, conf["dims"]];

  ok = Check[ Export[tgtPath, net, "ONNX"],
              Print["ERROR: Failed to export to ", tgtPath]; Return[$Failed]; ];
  ok
];

KeyValueMap[exportOne, specs];

Print["Done. Saved to ", outDir];
Exit[0];
