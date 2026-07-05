"""Generate a simple HTML inspection report for a single analysis."""

import base64
from datetime import datetime
from io import BytesIO

from PIL import Image

REPORT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Inspection Report – {filename}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 720px; margin: 2rem auto; color: #222; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: .5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  td, th {{ border: 1px solid #ccc; padding: .5rem .75rem; text-align: left; }}
  th {{ background: #f4f4f4; width: 220px; }}
  .verdict-FAIL {{ color: #b00020; font-weight: bold; }}
  .verdict-REVIEW {{ color: #b36b00; font-weight: bold; }}
  img {{ max-width: 320px; border: 1px solid #ccc; }}
</style>
</head>
<body>
<h1>Manufacturing Defect Inspection Report</h1>
<table>
  <tr><th>Image</th><td>{filename}</td></tr>
  <tr><th>Analyzed at</th><td>{timestamp}</td></tr>
  <tr><th>Predicted defect</th><td>{defect_title}</td></tr>
  <tr><th>Confidence</th><td>{confidence:.1%}</td></tr>
  <tr><th>Verdict</th><td class="verdict-{verdict}">{verdict}</td></tr>
  <tr><th>Defect description</th><td>{description}</td></tr>
  <tr><th>Recommended action</th><td>{recommendation}</td></tr>
</table>
<h2>Inspected image</h2>
<img src="data:image/png;base64,{image_b64}" alt="inspected image">
<p style="color:#888;font-size:.85rem">
Generated automatically by Manufacturing Defect Analyzer. AI predictions
require manual confirmation before final quality decisions.
</p>
</body>
</html>
"""


def generate_report(
    image: Image.Image,
    filename: str,
    defect_title: str,
    description: str,
    confidence: float,
    verdict: str,
    recommendation: str,
) -> str:
    """Render the inspection report and return it as an HTML string."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    return REPORT_TEMPLATE.format(
        filename=filename,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        defect_title=defect_title,
        description=description,
        confidence=confidence,
        verdict=verdict,
        recommendation=recommendation,
        image_b64=image_b64,
    )
