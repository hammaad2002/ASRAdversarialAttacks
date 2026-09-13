from torch import nn

LABELS = ["-", "A", "B", "C", "|"]


class TinyCTC(nn.Module):
    def __init__(self, vocab: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, vocab, kernel_size=3, padding=1)

    def forward(self, audio):
        hidden = self.conv(audio.unsqueeze(1))
        return hidden.transpose(1, 2), None
