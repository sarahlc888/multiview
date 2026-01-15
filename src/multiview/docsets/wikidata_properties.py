"""Wikidata property ID to human-readable name mapping.

This module provides mappings for common Wikidata properties (P-codes)
used in T-REx and other KG datasets.

Source: https://www.wikidata.org/wiki/Wikidata:List_of_properties
"""

# Most common Wikidata properties
# Format: P-code -> human-readable name
WIKIDATA_PROPERTIES = {
    # People & Biography
    "P19": "place of birth",
    "P20": "place of death",
    "P21": "sex or gender",
    "P22": "father",
    "P25": "mother",
    "P26": "spouse",
    "P27": "country of citizenship",
    "P40": "child",
    "P54": "member of sports team",
    "P69": "educated at",
    "P101": "field of work",
    "P102": "member of political party",
    "P103": "native language",
    "P106": "occupation",
    "P108": "employer",
    "P140": "religion",
    "P166": "award received",
    "P172": "ethnic group",
    "P185": "doctoral student",
    "P264": "record label",
    "P463": "member of",
    "P512": "academic degree",
    "P569": "date of birth",
    "P570": "date of death",
    "P734": "family name",
    "P735": "given name",
    # Organizations
    "P112": "founded by",
    "P127": "owned by",
    "P131": "located in administrative territory",
    "P155": "follows",
    "P156": "followed by",
    "P159": "headquarters location",
    "P169": "chief executive officer",
    "P176": "manufacturer",
    "P178": "developer",
    "P276": "location",
    "P355": "subsidiary",
    "P361": "part of",
    "P452": "industry",
    "P488": "chairperson",
    "P571": "inception",
    "P749": "parent organization",
    # Geography & Places
    "P17": "country",
    "P30": "continent",
    "P36": "capital",
    "P37": "official language",
    "P47": "shares border with",
    "P85": "anthem",
    "P122": "basic form of government",
    "P150": "contains administrative territory",
    "P190": "twinned administrative body",
    "P421": "located in time zone",
    "P610": "highest point",
    "P625": "coordinate location",
    "P901": "lowest point",
    # Creative Works
    "P50": "author",
    "P57": "director",
    "P86": "composer",
    "P110": "illustrator",
    "P123": "publisher",
    "P136": "genre",
    "P144": "based on",
    "P161": "cast member",
    "P170": "creator",
    "P175": "performer",
    "P179": "part of the series",
    "P272": "production company",
    "P275": "copyright license",
    "P291": "place of publication",
    "P344": "director of photography",
    "P364": "original language of work",
    "P407": "language of work or name",
    "P495": "country of origin",
    "P577": "publication date",
    "P580": "start time",
    "P582": "end time",
    "P585": "point in time",
    "P674": "characters",
    "P750": "distributor",
    "P800": "notable work",
    "P840": "narrative location",
    "P915": "filming location",
    "P921": "main subject",
    # Sports
    "P118": "league",
    "P641": "sport",
    "P1346": "winner",
    "P1350": "number of matches played",
    "P1440": "fencing weapon",
    # Science & Technology
    "P195": "collection",
    "P231": "CAS Registry Number",
    "P493": "ICD-9-CM",
    "P494": "ICD-10",
    "P527": "has part",
    "P1056": "product or material produced",
    # General Relations
    "P31": "instance of",
    "P105": "taxon rank",
    "P171": "parent taxon",
    "P279": "subclass of",
    "P373": "Commons category",
    "P460": "said to be the same as",
    "P461": "opposite of",
    "P910": "topic's main category",
    "P1343": "described by source",
    "P1411": "nominated for",
    "P1427": "start point",
    "P1559": "name in native language",
    "P1705": "native label",
}


def get_property_name(property_id: str) -> str:
    """Get human-readable name for a Wikidata property ID.

    Args:
        property_id: Wikidata property ID (e.g., "P106")

    Returns:
        Human-readable property name (e.g., "occupation")
        If not found, returns the original property_id
    """
    return WIKIDATA_PROPERTIES.get(property_id, property_id)
