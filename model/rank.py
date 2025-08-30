import torch
import torch.nn as nn

# Combined model with classifier for algorithm selection

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)  # Output same size as input for residual
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        return self.relu(out + residual)  # Apply activation after residual addition

class Rank(nn.Module):
    def __init__(self, model1, model2, num_classes=8, dropout_rate=0.3, input_dropout_rate=0.1):
        super(Rank, self).__init__()
        self.model1 = model1
        self.model2 = model2

        """ No Residual connections
        self.combiner = nn.Sequential(
            # We move the concatenation to the combiner
            # 3 linear layers with dropout and ReLU activation
            nn.Linear(320 + 96, 256), # 320 + 96 -> 128 + 128
            nn.ReLU(), ## nn.ReLU,
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16), 
            nn.ReLU(), ## nn.Sigmoid
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )
        """

        # MLP with residual connections
        # self.input_projection = nn.Linear(320 + 96, 256)  # Project to initial hidden size
            
        # Residual blocks
        # self.res_block1 = ResidualBlock(256, 512, dropout_rate)
        # self.fc1 = nn.Linear(256, 128)
        self.res_block20 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block21 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block22 = ResidualBlock(128, 256, dropout_rate=0.3)
        # Calculate dense connection input sizes
        # After 3 res blocks: 128 + 128 + 128 + 128 = 512
        self.fc2 = nn.Linear(512, 64)  # Updated from 128 to 512
        self.res_block30 = ResidualBlock(64, 128, dropout_rate=0.3)
        self.res_block31 = ResidualBlock(64, 128, dropout_rate=0.3)
        self.res_block32 = ResidualBlock(64, 128, dropout_rate=0.3)
        # After next 3 res blocks: 64 + 64 + 64 + 64 = 256  
        self.fc3 = nn.Linear(256, 32)  # Updated from 64 to 256

        # Add batch normalization
        # self.bn1 = nn.BatchNorm1d(256)
        # self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # self.input_dropout = nn.Dropout(input_dropout_rate)
            
        # Final classifier
        self.classifier = nn.Sequential(nn.Linear(32, num_classes), nn.Sigmoid())


    """No Residual connections
    def forward(self, data1, data2):
        # Get predictions from both models
        output1 = self.model1(data1)
        output2 = self.model2(data2)

        # Concatenate the outputs from both models
        combined_output = torch.cat((output1, output2), dim=1)

        # Apply the combiner to the concatenated outputs
        final_output = self.combiner(combined_output)

        return final_output
    """
    def forward(self, data1, data2):
        # Get predictions from both models
        output1 = self.model1(data1)
        output2 = self.model2(data2)
            
        # Concatenate the outputs from both models
        combined_output = torch.cat((output1, output2), dim=1)
            
        # Apply residual blocks with dense connections
        x1 = self.res_block20(combined_output)
        x2 = self.res_block21(x1)
        x3 = self.res_block22(x2)
    
        # Dense connection: concatenate all previous block outputs
        dense_input = torch.cat((combined_output, x1, x2, x3), dim=1)
    
        x = self.fc2(dense_input)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
    
        x4 = self.res_block30(x)
        x5 = self.res_block31(x4)
        x6 = self.res_block32(x5)
    
        # Another dense connection
        dense_input2 = torch.cat((x, x4, x5, x6), dim=1)
    
        x = self.fc3(dense_input2)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final classification
        final_output = self.classifier(x)
            
        return final_output