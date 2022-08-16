from lit_jupyter import JupyterLab
import lightning as L
import os
from components.grado import ImageServeGradio

class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.jupyter_work = JupyterLab(kernel=os.getenv("LIGHTNING_JUPYTER_LAB_KERNEL", "python"), cloud_compute=L.CloudCompute(os.getenv("LIGHTNING_JUPYTER_LAB_COMPUTE", "gpu")))
        self.serve_work = ImageServeGradio(L.CloudCompute("cpu"))

    def run(self):
        self.jupyter_work.run()

    def configure_layout(self):
        tab_1 = {"name": "JupyterLab", "content": self.jupyter_work}
        tab_2 = {"name": "Grado", "content": self.serve_work}
        tab_3 = {"name": "Paper", "content": "https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2014/MadhulikaUjjwal2013Drishti.pdf"}
        
        return [tab_1, tab_2, tab_3]

app = L.LightningApp(RootFlow())