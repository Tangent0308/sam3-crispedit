"""Protocol 2.0 labeling of external editing pairs (CrispEdit-2M, ScaleEdit).

One pipeline turns raw (BEFORE, AFTER, instruction) pairs into training records in the format of the
SAMTokEdit D3 (single unit) and D4 (multi unit) sets: protocol 2.0 instructions, mask semantics 2.0, AFTER
identical to BEFORE outside the edit, and a SAMTok token that names the source mask.

Steps (docs/EXTEDIT_PROTOCOL2.md):
  ingest    unify rows of both sources, write the image bytes as they are;
  precheck  CPU: size, registration, change map; drop no-op, global and misaligned pairs;
  plan      one Qwen3.8-27B call: is it a clean local edit, its units with protocol 2.0 slots and boxes;
  segment   SAM3 for every unit's S (add: the support, cut to the spot) and the SAMTok round trip;
  gate      edited region, edit gates, geometry, paste-back, post-paste checks;
  audit     render the ref / no-ref sentences, a second 27B call checks them against the images;
  export    D3 / D4 records, token cache, rejections, summary and review sheets.

Vendored byte for byte from Tangent0308/samtok_edit_benchmark 2.6.0 (bf2283d): instruction_protocol.py and
instruction_style.py.  The pixel, region and slot rules are ported from the code that produced D3 / D4
(samtok_edit v2 protocol_adapt "protocol-adapt-4", v3 protocol_v2_adapt, scripts/data_cleaning); each ported
function names its origin.
"""

PIPELINE_VERSION = "extedit-protocol2-1"
PROTOCOL_SOURCE = "Tangent0308/samtok_edit_benchmark@bf2283d (2.6.0):samtok_benchmark/v2/instruction_protocol.py"
PROTOCOL_SHA256 = "fe192e8d955a36ca58fb4a383585bb72110076f834d14294dd35e4ae286ffa85"
STYLE_SHA256 = "91456aeb1ad47c8a973bc5b7b9e532c436ef9200bb5870cc50acc8988c31dcee"
