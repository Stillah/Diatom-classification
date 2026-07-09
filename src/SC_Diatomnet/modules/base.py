from typing import Protocol
from abc import ABC, abstractmethod 



class SegmentModel(Protocol):

    @abstractmethod
    def segment(self, image):
        pass 


class ClassificationModel(Protocol):

    @abstractmethod
    def classify(self, image) -> str:
        pass
    