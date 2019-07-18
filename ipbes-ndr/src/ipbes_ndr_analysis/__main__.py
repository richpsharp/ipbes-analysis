import os
import sys
import logging

import ipbes_ndr_analysis
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        LOGGER.error(
            "usage: python %s iam_token_path workspace_dir", sys.argv[0])
        sys.exit(-1)
    raw_iam_token_path = sys.argv[1]
    raw_workspace_dir = sys.argv[2]
    if not os.path.isfile(raw_iam_token_path):
        LOGGER.error(
            '%s is not a file, should be an iam token', raw_workspace_dir)
        sys.exit(-1)
    if os.path.isfile(raw_workspace_dir):
        LOGGER.error(
            '%s is supposed to be the workspace directory but points to an '
            'existing file' % raw_workspace_dir)
        sys.exit(-1)
    ipbes_ndr_analysis.main(raw_iam_token_path, raw_workspace_dir)
