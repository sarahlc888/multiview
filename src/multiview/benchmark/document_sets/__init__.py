"""DocSet registry for benchmark tasks."""

from multiview.benchmark.document_sets.analogies import AnalogiesDocSet
from multiview.benchmark.document_sets.base import BaseDocSet
from multiview.benchmark.document_sets.crossword_clues import CrosswordCluesDocSet
from multiview.benchmark.document_sets.gsm8k import GSM8KDocSet
from multiview.benchmark.document_sets.hackernews import HackerNewsDocSet
from multiview.benchmark.document_sets.infinite_chats import InfiniteChatsDocSet
from multiview.benchmark.document_sets.infinite_prompts import InfinitePromptsDocSet
from multiview.benchmark.document_sets.rocstories import RocStoriesDocSet

DOCSETS = {
    "analogies": AnalogiesDocSet,
    "gsm8k": GSM8KDocSet,
    "crossword_clues": CrosswordCluesDocSet,
    "rocstories": RocStoriesDocSet,
    "hackernews": HackerNewsDocSet,
    "infinite_prompts": InfinitePromptsDocSet,
    "infinite_chats": InfiniteChatsDocSet,
}

__all__ = [
    "AnalogiesDocSet",
    "BaseDocSet",
    "DOCSETS",
    "GSM8KDocSet",
    "CrosswordCluesDocSet",
    "RocStoriesDocSet",
    "HackerNewsDocSet",
    "InfinitePromptsDocSet",
    "InfiniteChatsDocSet",
]
