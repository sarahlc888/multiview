"""DocSet registry for benchmark tasks."""

from multiview.docsets.abstractsim import AbstractSimDocSet
from multiview.docsets.aidanbench import AidanBenchDocSet
from multiview.docsets.analogies import AnalogiesDocSet
from multiview.docsets.arxiv_cs import ArxivCSDocSet
from multiview.docsets.base import BaseDocSet
from multiview.docsets.bills import BillsDocSet
from multiview.docsets.crossword_clues import CrosswordCluesDocSet
from multiview.docsets.d5 import D5DocSet
from multiview.docsets.d5_applic import D5ApplicabilityDocSet
from multiview.docsets.dickinson import DickinsonDocSet
from multiview.docsets.example_images import ExampleImagesDocSet
from multiview.docsets.fb15k237 import FB15k237DocSet
from multiview.docsets.feedbacks import FeedbacksClusteringDocSet
from multiview.docsets.fewevent import FewEventClusteringDocSet
from multiview.docsets.fewnerd import FewNerdClusteringDocSet
from multiview.docsets.fewrel import FewRelClusteringDocSet
from multiview.docsets.goodreads_quotes import GoodreadsQuotesDocSet
from multiview.docsets.gsm8k import GSM8KDocSet
from multiview.docsets.hackernews import HackerNewsDocSet
from multiview.docsets.haiku import HaikuDocSet
from multiview.docsets.infinite_chats import InfiniteChatsDocSet
from multiview.docsets.infinite_prompts import InfinitePromptsDocSet
from multiview.docsets.inspired import InspiredDocSet
from multiview.docsets.instructstsb import InstructSTSBDocSet
from multiview.docsets.intent_emotion import IntentEmotionDocSet
from multiview.docsets.moralfables import MoralFablesDocSet
from multiview.docsets.nytclustering import NYTClusteringDocSet
from multiview.docsets.onion_news import OnionNewsDocSet
from multiview.docsets.ratemyprof import RateMyProfClusteringDocSet
from multiview.docsets.rocstories import RocStoriesDocSet
from multiview.docsets.trex import TRExDocSet
from multiview.docsets.triz40 import TRIZ40DocSet
from multiview.docsets.wn18rr import WN18RRDocSet

DOCSETS = {
    "abstractsim": AbstractSimDocSet,
    "aidanbench": AidanBenchDocSet,
    "analogies": AnalogiesDocSet,
    "arxiv_cs": ArxivCSDocSet,
    "bills": BillsDocSet,
    "d5": D5DocSet,  # Kept for backwards compatibility
    "d5_doc2doc": D5DocSet,
    "d5_applicability": D5ApplicabilityDocSet,
    "dickinson": DickinsonDocSet,
    "example_images": ExampleImagesDocSet,
    "fb15k237": FB15k237DocSet,
    "goodreads_quotes": GoodreadsQuotesDocSet,
    "gsm8k": GSM8KDocSet,
    "haiku": HaikuDocSet,
    "crossword_clues": CrosswordCluesDocSet,
    "rocstories": RocStoriesDocSet,
    "hackernews": HackerNewsDocSet,
    "infinite_prompts": InfinitePromptsDocSet,
    "infinite_chats": InfiniteChatsDocSet,
    "inspired": InspiredDocSet,
    "intent_emotion": IntentEmotionDocSet,
    "instructstsb": InstructSTSBDocSet,
    "moralfables": MoralFablesDocSet,
    "nytclustering": NYTClusteringDocSet,
    "onion_news": OnionNewsDocSet,
    "ratemyprof": RateMyProfClusteringDocSet,
    "feedbacks": FeedbacksClusteringDocSet,
    "fewrel": FewRelClusteringDocSet,
    "fewnerd": FewNerdClusteringDocSet,
    "fewevent": FewEventClusteringDocSet,
    "trex": TRExDocSet,
    "triz40": TRIZ40DocSet,
    "wn18rr": WN18RRDocSet,
}

__all__ = [
    "AbstractSimDocSet",
    "AidanBenchDocSet",
    "AnalogiesDocSet",
    "ArxivCSDocSet",
    "BaseDocSet",
    "BillsDocSet",
    "D5DocSet",
    "D5ApplicabilityDocSet",
    "DickinsonDocSet",
    "DOCSETS",
    "ExampleImagesDocSet",
    "FB15k237DocSet",
    "GoodreadsQuotesDocSet",
    "GSM8KDocSet",
    "HaikuDocSet",
    "CrosswordCluesDocSet",
    "RocStoriesDocSet",
    "HackerNewsDocSet",
    "InfinitePromptsDocSet",
    "InfiniteChatsDocSet",
    "InspiredDocSet",
    "IntentEmotionDocSet",
    "InstructSTSBDocSet",
    "MoralFablesDocSet",
    "NYTClusteringDocSet",
    "OnionNewsDocSet",
    "RateMyProfClusteringDocSet",
    "FeedbacksClusteringDocSet",
    "FewRelClusteringDocSet",
    "FewNerdClusteringDocSet",
    "FewEventClusteringDocSet",
    "TRExDocSet",
    "TRIZ40DocSet",
    "WN18RRDocSet",
]
