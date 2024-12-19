import unittest
from unittest.mock import MagicMock
from sparselabel.processor import Processor


class TestProcessor(unittest.TestCase):

    def setUp(self):
        self.case_handler = MagicMock()
        self.case_loader = [MagicMock(), MagicMock()]
        self.processor = Processor(self.case_handler, self.case_loader)

    def test_process(self):
        self.processor.process()
        for case in self.case_loader:
            self.case_handler.apply.assert_any_call(case)
