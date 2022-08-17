import lightning as L
from components.gradio import ImageServeGradio

class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.gradio_work = ImageServeGradio(L.CloudCompute("cpu"))

    def run(self):
        self.gradio_work.run()

    def configure_layout(self):
        tab = {"name": "Grado EDA", "content": self.gradio_work}
        
        return [tab]

app = L.LightningApp(RootFlow())