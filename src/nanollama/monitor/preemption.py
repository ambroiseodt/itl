# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Preemption Handler

This file is useful to handle and requeue Slurm job that are being preempted.

@ 2025, Meta
"""

import logging
import os
import signal
from signal import SIGTERM, SIGUSR1, Signals
from types import FrameType, TracebackType

from ..distributed import get_hostname

logger = logging.getLogger("nanollama")


class PreemptionHandler:
    def __init__(self):
        self.preemption_flag = False

    def __enter__(self):
        def signal_handler(signum: Signals, frame: FrameType):
            logger.warning("Signal handler called with signal " + str(signum))
            self.preemption_flag = True

        def term_handler(signum: Signals, frame: FrameType):
            logger.warning("Received termination signal " + str(signum))
            # do not requeue to avoid requeuing on `scancel`

        signal.signal(SIGUSR1, signal_handler)
        signal.signal(SIGTERM, term_handler)
        return self

    def __call__(self):
        return self.preemption_flag

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        if self.preemption_flag:
            prod_id = int(os.environ["SLURM_PROCID"])
            logger.warning(f"Host: {get_hostname()} - Global rank: {prod_id}")
            if prod_id == 0:
                logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
                os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
            else:
                logger.warning("Not the master process, no need to requeue.")
