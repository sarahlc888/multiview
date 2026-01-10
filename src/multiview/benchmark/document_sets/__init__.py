"""DocSet registry for benchmark tasks."""

from multiview.benchmark.document_sets.analogies import AnalogiesDocSet
from multiview.benchmark.document_sets.base import BaseDocSet
from multiview.benchmark.document_sets.crossword_clues import CrosswordCluesDocSet
from multiview.benchmark.document_sets.gsm8k import GSM8KDocSet
from multiview.benchmark.document_sets.rocstories import RocStoriesDocSet

DOCSETS = {
    "analogies": AnalogiesDocSet,
    "gsm8k": GSM8KDocSet,
    "crossword_clues": CrosswordCluesDocSet,
    "rocstories": RocStoriesDocSet,
}

__all__ = [
    "AnalogiesDocSet",
    "BaseDocSet",
    "DOCSETS",
    "GSM8KDocSet",
    "CrosswordCluesDocSet",
    "RocStoriesDocSet",
]
