"""DocSet registry for benchmark tasks."""

from multiview.docsets.abstractsim import AbstractSimDocSet
from multiview.docsets.aidanbench import AidanBenchDocSet
from multiview.docsets.analogies import AnalogiesDocSet
from multiview.docsets.arxiv_abstract_sentences import ArxivAbstractSentencesDocSet
from multiview.docsets.arxiv_cs import ArxivCSDocSet
from multiview.docsets.base import BaseDocSet
from multiview.docsets.bills import BillsDocSet
from multiview.docsets.cari_aesthetics import CARIAestheticsDocSet
from multiview.docsets.crossword_clues import CrosswordCluesDocSet
from multiview.docsets.d5 import D5DocSet
from multiview.docsets.d5_applic import D5ApplicabilityDocSet
from multiview.docsets.dickinson import DickinsonDocSet
from multiview.docsets.example_images import ExampleImagesDocSet
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
from multiview.docsets.met_museum import MetMuseumDocSet
from multiview.docsets.mmlu import MMLUDocSet
from multiview.docsets.moralfables import MoralFablesDocSet
from multiview.docsets.new_yorker_cartoons import NewYorkerCartoonsDocSet
from multiview.docsets.newyorker_covers import NewYorkerCoversDocSet
from multiview.docsets.nytclustering import NYTClusteringDocSet
from multiview.docsets.onion_headlines import OnionNewsDocSet
from multiview.docsets.ratemyprof import RateMyProfClusteringDocSet
from multiview.docsets.rocstories import RocStoriesDocSet
from multiview.docsets.tao_te_ching import TaoTeChingDocSet
from multiview.docsets.trex import TRExDocSet
from multiview.docsets.triz40 import TRIZ40DocSet
from multiview.docsets.ut_zappos50k import UTZappos50kDocSet
from multiview.docsets.whole_earth_catalog import WholeEarthCatalogDocSet

DOCSETS = {
    "abstractsim": AbstractSimDocSet,
    "aidanbench": AidanBenchDocSet,
    "analogies": AnalogiesDocSet,
    "arxiv_cs": ArxivCSDocSet,
    "arxiv_abstract_sentences": ArxivAbstractSentencesDocSet,
    "bills": BillsDocSet,
    "cari_aesthetics": CARIAestheticsDocSet,
    "d5": D5DocSet,  # Kept for backwards compatibility
    "d5_doc2doc": D5DocSet,
    "d5_applicability": D5ApplicabilityDocSet,
    "dickinson": DickinsonDocSet,
    "example_images": ExampleImagesDocSet,
    "goodreads_quotes": GoodreadsQuotesDocSet,
    "gsm8k": GSM8KDocSet,
    "haiku": HaikuDocSet,
    "crossword_clues": CrosswordCluesDocSet,
    "rocstories": RocStoriesDocSet,
    "hackernews": HackerNewsDocSet,
    "infinite_prompts": InfinitePromptsDocSet,
    "infinite_chats": InfiniteChatsDocSet,
    "inspired": InspiredDocSet,
    "inb_intent_emotion": IntentEmotionDocSet,
    "inb_instructstsb": InstructSTSBDocSet,
    "met_museum": MetMuseumDocSet,
    "mmlu": MMLUDocSet,
    "new_yorker_cartoons": NewYorkerCartoonsDocSet,
    "newyorker_covers": NewYorkerCoversDocSet,
    "moralfables": MoralFablesDocSet,
    "inb_nytclustering": NYTClusteringDocSet,
    "onion_headlines": OnionNewsDocSet,
    "inb_ratemyprof": RateMyProfClusteringDocSet,
    "inb_feedbacks": FeedbacksClusteringDocSet,
    "inb_fewrel": FewRelClusteringDocSet,
    "inb_fewnerd": FewNerdClusteringDocSet,
    "inb_fewevent": FewEventClusteringDocSet,
    "tao_te_ching": TaoTeChingDocSet,
    "trex": TRExDocSet,
    "triz40": TRIZ40DocSet,
    "ut_zappos50k": UTZappos50kDocSet,
    "whole_earth_catalog": WholeEarthCatalogDocSet,
}

__all__ = [
    "AbstractSimDocSet",
    "AidanBenchDocSet",
    "AnalogiesDocSet",
    "ArxivCSDocSet",
    "ArxivAbstractSentencesDocSet",
    "BaseDocSet",
    "BillsDocSet",
    "CARIAestheticsDocSet",
    "D5DocSet",
    "D5ApplicabilityDocSet",
    "DickinsonDocSet",
    "DOCSETS",
    "ExampleImagesDocSet",
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
    "MetMuseumDocSet",
    "MMLUDocSet",
    "NewYorkerCartoonsDocSet",
    "NewYorkerCoversDocSet",
    "MoralFablesDocSet",
    "NYTClusteringDocSet",
    "OnionNewsDocSet",
    "RateMyProfClusteringDocSet",
    "FeedbacksClusteringDocSet",
    "FewRelClusteringDocSet",
    "FewNerdClusteringDocSet",
    "FewEventClusteringDocSet",
    "TaoTeChingDocSet",
    "TRExDocSet",
    "TRIZ40DocSet",
    "UTZappos50kDocSet",
    "WholeEarthCatalogDocSet",
]
