import lightning as L
from components.grado import ImageServeGradio

class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.grado_work = ImageServeGradio(L.CloudCompute("cpu"))

    def run(self):
        self.grado_work.run()

    def configure_layout(self):
        tab = {"name": "Grado EDA", "content": self.grado_work}
        
        return [tab]

app = L.LightningApp(RootFlow())