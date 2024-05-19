import torch


def bytesToTensor(frame,
                  half,
                  bf16,
                  width,
                  height) -> torch.Tensor:
    frame = torch.frombuffer(frame, dtype=torch.uint8).reshape(height,width,3)
    
    if half and not bf16:
        frame = (
            torch.tensor
            (
                frame
            ).permute(2, 0, 1).unsqueeze(0).half().mul_(1 / 255)
        )
    elif  bf16:
        frame = (
            torch.tensor
            (
                frame
            ).permute(2, 0, 1).unsqueeze(0).bfloat16().mul_(1 / 255)
        )     
    else:
        frame = (
            torch.tensor
            (
                frame
            ).permute(2, 0, 1).unsqueeze(0).float().mul_(1 / 255)
        )
    return frame