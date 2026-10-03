developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ CANNBOT_BIN=/home/developer/.nvm/versions/node/v18.19.1/bin/cannbot
mp/cannbot-skill-check.XXXXXX)

"$CANNBOT_BIN" debug skill > "$CANNBOT_CHECK_DIR/skills.json" 2> "$Cdeveloper@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ CANNBOT_CHECK_DIR=$(mktemp -d /tmp/cannbot-skill-check.XXXXXX)
ANNBOT_CHECK_DIR/stderr.log"

printf 'CLI退出码：%s\n检查目录：%s\n' "$?" "$CANNBOT_CHECK_DIR"

python3 - "$CANNBOT_CHECK_DIR/skills.json" <<'PY'
import json, developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ 
sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
print(f"输出文件：{path}，大developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ "$CANNBOT_BIN" debug skill > "$CANNBOT_CHECK_DIR/skills.json" 2> "$CANNBOT_CHECK_DIR/stderr.log"
小：{len(raw)} 字节")

skills = json.loads(raw.decode("utf-8-sig"))
found = [
    s for s in skills
    if s.get("name", "").startswith("triton-")
    or s.get("name") == "npu-arch"
]

for s in found:
    print(s["name"], "->", s.get("location"))

if not found:
    print("未发现 Triton 专属技能")
PY
developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ 
developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ printf 'CLI退出码：%s\n检查目录：%s\n' "$?" "$CANNBOT_CHECK_DIR"
CLI退出码：0
检查目录：/tmp/cannbot-skill-check.DYxSYW
developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ 
developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ python3 - "$CANNBOT_CHECK_DIR/skills.json" <<'PY'
> import json, sys
> from pathlib import Path
>
> path = Path(sys.argv[1])
> raw = path.read_bytes()
> print(f"输出文件：{path}，大小：{len(raw)} 字节")
>
> skills = json.loads(raw.decode("utf-8-sig"))
> found = [
>  s for s in skills
>  if s.get("name", "").startswith("triton-")
>  or s.get("name") == "npu-arch"
> ]
>
> for s in found:
>  print(s["name"], "->", s.get("location"))
>
> if not found:
>  print("未发现 Triton 专属技能")
> PY
> 输出文件：/tmp/cannbot-skill-check.DYxSYW/skills.json，大小：156061 字节
> 未发现 Triton 专属技能