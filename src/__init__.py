"""Root package information for the CLPT library."""

__name__ = 'clpt'
__version__ = '0.0.1'
__author__ = ['Will Buchanan', 'Saranya Krishnamoorthy', 'John Ortega', 'Ayush Singh', 'Yanyi Jiang']
__author_email__ = ['wbuchanan1@evicore.com', 'Saranya.Krishnamoorthy@evicore.com', 'John.Ortega@evicore.com',
                    'Ayush.Singh@evicore.com', 'Yanyi.Jiang@evicore.com']
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2022, {__author__}.'
__homepage__ = 'https://inqbator-gitlab.innovate.lan/ai/clpt'
__docs__ = "xxx"

import logging

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler())
_logger.setLevel(logging.INFO)
