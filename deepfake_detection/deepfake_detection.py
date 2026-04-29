import torch
import torch.nn as nn
import timm
from torchvision import models

# MEMBER 2: Video Pipeline (EfficientNet-B4 + ViT-Small)
class VideoPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches Report: EfficientNet-B4 spatial extractor [cite: 235]
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        # Matches Report: 4-layer ViT-Small for temporal modeling [cite: 236, 237]
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1792, nhead=8, batch_first=True), 
            num_layers=4
        )
        self.fc = nn.Linear(1792, 1)

    def forward(self, x):
        batch, frames, c, h, w = x.shape
        x = x.view(batch * frames, c, h, w)
        spatial = self.backbone(x).view(batch, frames, -1)
        temporal = self.temporal(spatial)
        return self.fc(temporal[:, -1, :])

# MEMBER 3: Audio Pipeline (ResNet-18)
class AudioPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches Report: ResNet-18 for spectrograms [cite: 238]
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x)

# MEMBER 4: Late Fusion System
class DeepfakeDetectionSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_model = VideoPipeline()
        self.audio_model = AudioPipeline()
        # Matches Report: Logistic regression meta-learner [cite: 239]
        self.meta_learner = nn.Linear(2, 1)

    def forward(self, video, audio):
        v_score = torch.sigmoid(self.video_model(video))
        a_score = torch.sigmoid(self.audio_model(audio))
        
        # Late Fusion strategy allows modality independence [cite: 191, 220]
        combined = torch.cat((v_score, a_score), dim=1)
        return torch.sigmoid(self.meta_learner(combined)), v_score, a_score
