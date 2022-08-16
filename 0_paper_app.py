import lightning as L

class RootFlow(L.LightningFlow):

    def configure_layout(self):
        tab_0 = {"name": "Paper", "content": "https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2014/MadhulikaUjjwal2013Drishti.pdf"}
        return [tab_0]

app = L.LightningApp(RootFlow())