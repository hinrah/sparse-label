from logging_config import logger
from check_datasets import DataErrorNotAllDataUsable, DataErrorThatLeadsToWrongLabels


class DatasetTester:
    def __init__(self, tests):
        self._tests = tests

    def apply(self, case):
        for test in self._tests:
            try:
                test.check(case)
            except DataErrorThatLeadsToWrongLabels as e:
                logger.warning(str(e) + ' This can lead to wrong labels that are not "unknown"')
            except DataErrorNotAllDataUsable as e:
                logger.info(str(e) + ' This CrossSection will be ignored and not used for training')
