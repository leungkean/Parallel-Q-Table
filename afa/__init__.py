# Register Datasets
import afa.datasets.adult
import afa.datasets.bsds
import afa.datasets.cube
import afa.datasets.gas
import afa.datasets.gas10
import afa.datasets.hepmass
import afa.datasets.miniboone
import afa.datasets.mnist16
import afa.datasets.power
import afa.datasets.molecule
import afa.datasets.molecule_small
import afa.datasets.molecule_20
import afa.datasets.molecule_10
import afa.datasets.molecule_15
import afa.datasets.psych_Compulsivity
import afa.datasets.psych_DetatchDay
import afa.datasets.psych_Exhibitionism
import afa.datasets.psych_HostileDay
import afa.datasets.psych_ImpulsivityDay
import afa.datasets.psych_ManipDay
import afa.datasets.psych_NegAffDay
import afa.datasets.psych_PsychoDay
import afa.datasets.psych_Urgency
import afa.datasets.psych_Total


# The below code is a workaround to disable a warning message that is coming from
# TensorFlow Probability and dm-tree. Eventually, this code can probably be removed,
# once the warning has been addressed in those packages.
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
