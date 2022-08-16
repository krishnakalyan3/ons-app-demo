from lit_jupyter import JupyterLab
import lightning as L
import os

class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.jupyter_work = JupyterLab(kernel=os.getenv("LIGHTNING_JUPYTER_LAB_KERNEL", "python"), cloud_compute=L.CloudCompute(os.getenv("LIGHTNING_JUPYTER_LAB_COMPUTE", "gpu")))

    def run(self):
        self.jupyter_work.run()

    def configure_layout(self):
        tab = {"name": "JupyterLab", "content": self.jupyter_work}
        return [tab]

app = L.LightningApp(RootFlow())