"""Agent version constants.

Bump the relevant version whenever agent behaviour changes — prompt edits,
new output fields, changed retrieval logic, etc. — so saved extractions and
compliance reports are always traceable back to the exact agent that produced
them.

Versioning scheme: MAJOR.MINOR.PATCH
  MAJOR — breaking change to output schema or fundamental approach
  MINOR — new fields, improved prompts, new capabilities
  PATCH — bug fixes, wording tweaks, performance tuning with no schema change
"""

# Pitch-deck extractor (extractor.py + deck_context.py)
EXTRACTOR_VERSION = "1.2.0"
# 1.0.0 — initial release
# 1.1.0 — added `industry` field to CompanyIdentity; improved INDUSTRY FIELD
#          prompt guidance for conservative inference
# 1.2.0 — local Ollama model support (pypdf text extraction path)

# Compliance analyzer (analyzer.py + agent.py)
ANALYZER_VERSION = "1.3.0"
# 1.0.0 — initial release; SEC dense-retrieval path
# 1.1.0 — web_search path for market/industry claims when no CIK
# 1.2.0 — streaming (iter_compliance_report); per-claim web_sources URLs
# 1.3.0 — parallel claim analysis (ThreadPoolExecutor); capped thinking budget
