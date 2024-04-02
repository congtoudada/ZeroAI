from abc import ABC
from typing import List

from zero.core.component.base.component import Component


class Pipline(ABC):

    def __init__(self):
        self.enable = True
        self.children: List[Component] = []

    def add_component(self, comp: Component):
        self.children.append(comp)

    def start(self):
        pass

    def pause(self):
        self.enable = False

    def update(self) -> bool:
        if not self.enable:
            return False
        else:
            return True

    def resume(self):
        self.enable = True

    def destroy(self):
        pass

