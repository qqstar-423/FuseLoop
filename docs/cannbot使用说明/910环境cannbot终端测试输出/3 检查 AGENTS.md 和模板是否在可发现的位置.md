developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ python3 - <<'PY'
> import json
>
> with open("/tmp/cannbot-skill-check.DYxSYW/skills.json", encoding="utf-8-sig") as f:
>  skills = json.load(f)
>
> print("发现技能数量：", len(skills))
> for s in skills:
>  print(s.get("name"), "->", s.get("location"))
> PY
> 发现技能数量： 10
> customize-opencode -> <built-in>
> markdown-filename-quality-checker -> /home/developer/.config/opencode/skills/markdown-filename-quality-checker/SKILL.md
> api-doc-qc-aclnn -> /home/developer/.config/opencode/skills/api-doc-qc-aclnn/SKILL.md
> ascendc-doc-gen -> /home/developer/.config/opencode/skills/ascendc-doc-gen/SKILL.md
> ascendc-markdown-quality-checker -> /home/developer/.config/opencode/skills/ascendc-markdown-quality-checker/SKILL.md
> markdown-punctuation-quality-checker -> /home/developer/.config/opencode/skills/markdown-punctuation-quality-checker/SKILL.md
> ascendc-api-retrieval -> /home/developer/.config/opencode/skills/ascendc-api-retrieval/SKILL.md
> general-markdown-quality-checker -> /home/developer/.config/opencode/skills/general-markdown-quality-checker/SKILL.md
> aclnn-doc-gen -> /home/developer/.config/opencode/skills/aclnn-doc-gen/SKILL.md
> markdown-compliance-quality-checker -> /home/developer/.config/opencode/skills/markdown-compliance-quality-checker/SKILL.md