import torch
import torch.nn as nn

# Combined model with classifier for algorithm selection
class Point(nn.Module):
    def __init__(self, model1, model2, num_classes=8, dropout_rate=0.3):
        super(Point, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.combiner = nn.Sequential(
            nn.Linear(num_classes * 2, 16),
            nn.Sigmoid(), ## nn.ReLU
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, data1, data2):
        # Get predictions from both models
        output1 = self.model1(data1)
        output2 = self.model2(data2)

        # Concatenate the outputs from both models
        combined_output = torch.cat((output1, output2), dim=1)

        # Apply the combiner to the concatenated outputs
        final_output = self.combiner(combined_output)

        return final_output
    
# 