# Required Libraries
import torch
import torch.nn as nn

class LearnableFusion(nn.Module):
    def __init__(self, cnn_output_size, rnn_output_size, fusion_output_size):
        super(LearnableFusion, self).__init__()
        # Resizing layers for cnn and rnn outputs
        self.cnn_resize = nn.Linear(cnn_output_size, max(cnn_output_size, rnn_output_size))
        self.rnn_resize = nn.Linear(rnn_output_size, max(cnn_output_size, rnn_output_size))
        # Fully connected fusion layer
        self.fc = nn.Linear(max(cnn_output_size, rnn_output_size) * 2, fusion_output_size)

    def forward(self, cnn_output, rnn_output):
        # Ensure the batch sizes are the same
        if cnn_output.size(0) != rnn_output.size(0):
            min_batch_size = min(cnn_output.size(0), rnn_output.size(0))
            cnn_output = cnn_output[:min_batch_size]
            rnn_output = rnn_output[:min_batch_size]
        
        # Resize both outputs to the same feature size
        cnn_output = self.cnn_resize(cnn_output)
        rnn_output = self.rnn_resize(rnn_output)
        
        # Debug: Print tensor shapes
        # print(f"cnn_output shape after resize: {cnn_output.shape}")
        # print(f"rnn_output shape after resize: {rnn_output.shape}")
        
        # Concatenate the resized outputs along the feature dimension
        fused_output = torch.cat((cnn_output, rnn_output), dim=1)  # Concatenate along the feature dimension
        
        # Debug: Print shape after concatenation
        # print(f"fused_output shape after concatenation: {fused_output.shape}")
        
        # Pass through the fully connected fusion layer
        fused_output = self.fc(fused_output)
        return fused_output