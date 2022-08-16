import lightning as L
from lightning_app.components.python import PopenPythonScript
from pathlib import Path


class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train_script = "glaucoma-classification.py"
        self.train_work = PopenPythonScript(Path(__file__).parent / self.train_script, cloud_compute=L.CloudCompute("gpu"))

    def run(self):
        self.train_work.run()
        self._exit("Application End!")

app = L.LightningApp(RootFlow())