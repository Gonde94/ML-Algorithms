import torch
import torch.nn as nn
import tiktoken

tokeniser = tiktoken.get_encoding("gpt2")

# Utility function to calculate the cross entropy loss of a given batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


# And now a function to compute the loss over all the batches sampled by a data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) # iterate over all batches if no fixed batch size given
    else:
        num_batches = min(num_batches, len(data_loader)) 
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item() # sum loss for each batch
        else:
            break
    return total_loss / num_batches # average the loss over all the batches


def text_to_token_ids(text, tokeniser):
    encoded = tokeniser.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adds batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokeniser):
    flat = token_ids.squeeze(0) # removes batch dimension
    return tokeniser.decode(flat.tolist())