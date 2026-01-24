---
name: prevent-verbose-responses
enabled: true
event: stop
conditions:
  - field: response_length
    operator: greater_than
    pattern: "2500"
action: warn
---

⚠️ **WARNING: Verbose Response Detected**

Your response is lengthy and may violate conciseness requirements:

**RESPONSE LENGTH VIOLATION:**
- Current response exceeds 2500 characters
- Non-code responses should be MAXIMUM 3-4 paragraphs
- Responses should focus on key findings with actionable next steps

**COMMON VERBOSITY ISSUES:**
- Repeating previously established context
- Using phrases like "as I mentioned", "previously", "to recap"
- Adding unnecessary explanations or background
- Multiple paragraphs when bullet points would work
- Describing what you'll do instead of just doing it

**REQUIRED FORMAT:**
- **Hard limit:** 3-4 paragraphs maximum for non-code answers
- **Focus:** Only actionable information and new insights
- **Style:** Direct, concise, no unnecessary elaboration
- **Exception:** Code examples, diffs, and technical output are exempt

**QUICK FIXES:**
- Remove repetitive phrases and context recap
- Combine related points into fewer paragraphs
- Use bullet points for lists instead of prose
- Focus on "what" and "next steps" rather than "why" explanations

**REQUIRED ACTION:**
Review and condense your response to essential information only.

**Remember:** Concise summaries with actionable information ONLY - no verbose explanations.