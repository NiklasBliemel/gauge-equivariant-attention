from modules.TransformerModules import Transformer
from modules.Preconditioner import transformer
from modules.TrainingFunctions import DwcTrainer
from modules.Constants import *

structure = (GAUGE_FIELD_SMALL, NON_GAUGE_DOF, 16, True)
module = transformer("Tr_4_16_True.pth")
# module = Transformer(*structure)

dwc_trainer = DwcTrainer(module, structure)

dwc_trainer.train(small=False)

dwc_trainer.safe_data("Tr")
