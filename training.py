import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import zipfile
from IPython.display import FileLink

# Importar del dataset y del modelo local
from data import Wikipedia, tokens_tensor, tokenizer
from model import Config, GPT2

def create_link(dir, name):
    full_path = os.path.join(dir, name)
    if os.path.exists(full_path):
        relative_path = os.path.relpath(full_path)
        print("Haz clic en el siguiente enlace para descargar tu checkpoint:")
        display(FileLink(relative_path))
    else:
        print(f"No se encontró el archivo en: {full_path}")

def remove_prev_checkpoint(save_dir, name):
    prev_checkpoint_path = os.path.join(save_dir, name)
    zip_prev_checkpoint_path = prev_checkpoint_path[:-3] + '.zip'
    if os.path.exists(prev_checkpoint_path) and os.path.exists(zip_prev_checkpoint_path):
        print(f"Previous checkpoint '{prev_checkpoint_path}' found. Deleting...")
        os.remove(zip_prev_checkpoint_path)
        os.remove(prev_checkpoint_path)
        print(f"Previous checkpoint '{prev_checkpoint_path}' removed.")

def save_and_zip(save_dir, checkpoint):
    checkpoint_path = os.path.join(save_dir, f"gpt2_step_{checkpoint['step']}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"--> Checkpoint guardado en: {checkpoint_path} (Loss: {checkpoint['loss']:.4f})")
    zip_filename = checkpoint_path[:-3] + '.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(checkpoint_path, os.path.basename(checkpoint_path))
    print(f"Archivo '{checkpoint_path}' comprimido a '{zip_filename}' con éxito")

def create_dataloader(dataset, from_step=0, order=None, batch_size=16):
    idxs = torch.arange(from_step, len(dataset))
    sub_dataset = Subset(dataset, idxs)
    if order is None:
        dataloader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False)
    else:
        sampler = SubsetRandomSampler(order[from_step:])
        dataloader = DataLoader(sub_dataset, batch_size=batch_size, sampler=sampler)
    return dataloader

def train(model, dataset, optimizer, device, epochs=2, batch_size=16, order=None, save_dir="/content/mi_transformer/params/", from_step=0):
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    dataloader = create_dataloader(dataset, from_step, order, batch_size)
    global_step = from_step

    for epoch in range(epochs):
        if global_step > 0 and epoch == 0:
            if order is None:
                sampler = list(range(from_step, len(dataset)))
            else:
                sampler = order[from_step:]
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            if order is None:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            else:
                sampler = order[from_step:]
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 500 == 0:
                checkpoint = {
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }
                save_and_zip(save_dir, checkpoint)
                remove_prev_checkpoint(save_dir, f"gpt2_step_{global_step - 500}.pt")
                create_link(save_dir, f"gpt2_step_{global_step}.zip")

            if global_step % 50 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")

if __name__ == '__main__':
    dataset = Wikipedia(tokens_tensor)
    vocab_size = tokenizer.vocab_size
    max_seq_length = 128
    seq_length = 128
    d_model = 768
    num_layers = 12
    num_heads = 12
    d_ff = d_model * 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = Config(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        seq_length=seq_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device,
        dropout=0.1
    )

    model = GPT2(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Iniciando entrenamiento desde training.py...")
    train(model=model,
          dataset=dataset,
          optimizer=optimizer,
          device=device,
          epochs=4,
          batch_size=16,
          order=None,
          save_dir=os.path.join(repo_path, 'params'),
          from_step=0
    )
