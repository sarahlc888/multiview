"""DocSet registry for benchmark tasks."""

from multiview.docsets.abstractsim import AbstractSimDocSet
from multiview.docsets.analogies import AnalogiesDocSet
from multiview.docsets.base import BaseDocSet
from multiview.docsets.crossword_clues import CrosswordCluesDocSet
from multiview.docsets.d5 import D5DocSet
from multiview.docsets.dickinson import DickinsonDocSet
from multiview.docsets.gsm8k import GSM8KDocSet
from multiview.docsets.hackernews import HackerNewsDocSet
from multiview.docsets.infinite_chats import InfiniteChatsDocSet
from multiview.docsets.infinite_prompts import InfinitePromptsDocSet
from multiview.docsets.moralfables import MoralFablesDocSet
from multiview.docsets.rocstories import RocStoriesDocSet

DOCSETS = {
    "abstractsim": AbstractSimDocSet,
    "analogies": AnalogiesDocSet,
    "d5": D5DocSet,
    "dickinson": DickinsonDocSet,
    "gsm8k": GSM8KDocSet,
    "crossword_clues": CrosswordCluesDocSet,
    "rocstories": RocStoriesDocSet,
    "hackernews": HackerNewsDocSet,
    "infinite_prompts": InfinitePromptsDocSet,
    "infinite_chats": InfiniteChatsDocSet,
    "moralfables": MoralFablesDocSet,
}

__all__ = [
    "AbstractSimDocSet",
    "AnalogiesDocSet",
    "BaseDocSet",
    "D5DocSet",
    "DickinsonDocSet",
    "DOCSETS",
    "GSM8KDocSet",
    "CrosswordCluesDocSet",
    "RocStoriesDocSet",
    "HackerNewsDocSet",
    "InfinitePromptsDocSet",
    "InfiniteChatsDocSet",
    "MoralFablesDocSet",
]
