import lightning as L
from lightning_app.components.python import PopenPythonScript
from components.grado import ImageServeGradio
from pathlib import Path


class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train_script = "glaucoma-classification.py"
        self.train_work = PopenPythonScript(script_path=Path(__file__).parent / self.train_script, 
                                cloud_compute=L.CloudCompute("gpu"))
        self.grado_work = ImageServeGradio(cloud_compute=L.CloudCompute("cpu"), parallel=False)

    def run(self):
        self.train_work.run()
        self.grado_work.run()

    def configure_layout(self):
        tab = {"name": "Demo", "content": self.grado_work}
        return [tab]

app = L.LightningApp(RootFlow())