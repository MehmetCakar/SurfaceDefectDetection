"""Defect descriptions and recommendation logic for the inspection app."""

DEFECT_INFO = {
    "crazing": {
        "title": "Crazing",
        "description": (
            "A network of fine surface cracks, usually caused by thermal "
            "stresses during rolling or cooling."
        ),
    },
    "inclusion": {
        "title": "Inclusion",
        "description": (
            "Non-metallic particles (oxides, slag) embedded in the steel "
            "surface during the manufacturing process."
        ),
    },
    "patches": {
        "title": "Patches",
        "description": (
            "Irregular surface regions with different texture or color, often "
            "caused by uneven cooling or contamination."
        ),
    },
    "pitted_surface": {
        "title": "Pitted Surface",
        "description": (
            "Small holes or cavities in the surface, typically caused by "
            "localized corrosion or trapped gases."
        ),
    },
    "rolled-in_scale": {
        "title": "Rolled-in Scale",
        "description": (
            "Oxide scale pressed into the surface during hot rolling, leaving "
            "dark irregular marks."
        ),
    },
    "scratches": {
        "title": "Scratches",
        "description": (
            "Linear surface damage caused by mechanical contact with rollers, "
            "guides or handling equipment."
        ),
    },
}


def recommendation(confidence: float) -> tuple[str, str]:
    """Map confidence to (verdict, recommendation text)."""
    if confidence > 0.85:
        return (
            "FAIL",
            "High confidence defect classification. Manual confirmation "
            "recommended before final rejection.",
        )
    if confidence >= 0.60:
        return (
            "REVIEW",
            "Moderate confidence. Further visual inspection required.",
        )
    return (
        "REVIEW",
        "Low confidence. Image quality or defect visibility may be "
        "insufficient — retake the image or inspect manually.",
    )
