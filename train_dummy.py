# train_dummy.py
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data.data import TrainDataset
from models.network import Generator, Discriminator
from logs.log import TensorBoardX
from utils.utils import set_requires_grad, save_model, save_optimizer, resume_model, resume_optimizer

# Load konfigurasi dummy
from config.config_dummy import *
config = {
    'train': train,
    'G': G,
    'D': D,
    'loss': loss,
    'feature_extract_model': feature_extract_model
}

test_time = False

if __name__ == "__main__":
    with open(config['train']['img_list'], 'r') as f:
        img_list = [line.strip() for line in f if line.strip()]

    dataloader = DataLoader(
        TrainDataset(img_list),
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(
        zdim=config['G']['zdim'],
        use_batchnorm=config['G']['use_batchnorm'],
        use_residual_block=config['G']['use_residual_block'],
        num_classes=config['G']['num_classes']
    ).to(device)
    D = Discriminator(use_batchnorm=config['D']['use_batchnorm']).to(device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=config['train']['learning_rate'])
    optimizer_D = torch.optim.Adam(D.parameters(), lr=config['train']['learning_rate'])
    last_epoch = -1

    if config['train']['resume_model']:
        e1 = resume_model(G, config['train']['resume_model'])
        e2 = resume_model(D, config['train']['resume_model'])
        assert e1 == e2
        last_epoch = e1

    if config['train']['resume_optimizer']:
        e3 = resume_optimizer(optimizer_G, G, config['train']['resume_optimizer'])
        e4 = resume_optimizer(optimizer_D, D, config['train']['resume_optimizer'])
        assert e3 == e4 == last_epoch

    tb = TensorBoardX(config_filename_list=["config_dummy.py"])

    l1_loss = torch.nn.L1Loss().to(device)

    for epoch in range(last_epoch + 1, config['train']['num_epochs']):
        t = time.time()
        for step, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)

            z = torch.FloatTensor(np.random.uniform(-1, 1, (len(batch['img']), config['G']['zdim']))).to(device)

            # ============ Train Discriminator ============
            img128_fake, img64_fake, img32_fake, _, _, _, _, _, _, _ = G(
                batch['img'], batch['img64'], batch['img32'],
                batch['left_eye'], batch['right_eye'], batch['nose'], batch['mouth'], z, use_dropout=True
            )

            set_requires_grad(D, True)
            adv_D_loss = -torch.mean(D(batch['img_frontal'])) + torch.mean(D(img128_fake.detach()))
            alpha = torch.rand(batch['img_frontal'].shape[0], 1, 1, 1).expand_as(batch['img_frontal']).to(device)
            interpolated = Variable(alpha * img128_fake.detach() + (1 - alpha) * batch['img_frontal'], requires_grad=True)
            out = D(interpolated)
            grad = torch.autograd.grad(
                outputs=out,
                inputs=interpolated,
                grad_outputs=torch.ones_like(out),
                retain_graph=True,
                create_graph=True,
                only_inputs=True
            )[0].view(out.shape[0], -1)
            gp_loss = ((grad.norm(2, dim=1) - 1) ** 2).mean()
            L_D = adv_D_loss + config['loss']['weight_gradient_penalty'] * gp_loss
            optimizer_D.zero_grad()
            L_D.backward()
            optimizer_D.step()

            # ============ Train Generator ============
            set_requires_grad(D, False)
            adv_G_loss = -torch.mean(D(img128_fake))
            pixelwise_loss = (
                config['loss']['weight_128'] * l1_loss(img128_fake, batch['img_frontal']) +
                config['loss']['weight_64'] * l1_loss(img64_fake, batch['img64_frontal']) +
                config['loss']['weight_32'] * l1_loss(img32_fake, batch['img32_frontal'])
            )

            # Flip symmetry loss
            img128_flip = img128_fake.flip(dims=[3])
            img64_flip = img64_fake.flip(dims=[3])
            img32_flip = img32_fake.flip(dims=[3])
            symmetry_loss = (
                config['loss']['weight_128'] * l1_loss(img128_fake, img128_flip) +
                config['loss']['weight_64'] * l1_loss(img64_fake, img64_flip) +
                config['loss']['weight_32'] * l1_loss(img32_fake, img32_flip)
            )

            tv_loss = (
                torch.mean(torch.abs(img128_fake[:, :, :-1, :] - img128_fake[:, :, 1:, :])) +
                torch.mean(torch.abs(img128_fake[:, :, :, :-1] - img128_fake[:, :, :, 1:]))
            )

            L_G = (
                config['loss']['weight_pixelwise'] * pixelwise_loss +
                config['loss']['weight_symmetry'] * symmetry_loss +
                config['loss']['weight_adv_G'] * adv_G_loss +
                config['loss']['weight_total_varation'] * tv_loss
            )

            optimizer_G.zero_grad()
            L_G.backward()
            optimizer_G.step()

            # Logging
            tb.add_scalar("D_loss", L_D.item(), epoch * len(dataloader) + step, 'train')
            tb.add_scalar("G_loss", L_G.item(), epoch * len(dataloader) + step, 'train')
            tb.add_scalar("pixelwise_loss", pixelwise_loss.item(), epoch * len(dataloader) + step, 'train')
            tb.add_scalar("symmetry_loss", symmetry_loss.item(), epoch * len(dataloader) + step, 'train')
            tb.add_scalar("tv_loss", tv_loss.item(), epoch * len(dataloader) + step, 'train')

            if step % config['train']['log_step'] == 0:
                print(f"Epoch {epoch}, Step {step}, D_loss: {L_D.item():.4f}, G_loss: {L_G.item():.4f}")
                tb.add_image_grid("grid/predict", 4, img128_fake.data.float() / 2.0 + 0.5, epoch * len(dataloader) + step, 'train')
                tb.add_image_grid("grid/frontal", 4, batch['img_frontal'].data.float() / 2.0 + 0.5, epoch * len(dataloader) + step, 'train')

        # Save checkpoint setiap akhir epoch
        save_model(G, tb.path, epoch)
        save_model(D, tb.path, epoch)
        save_optimizer(optimizer_G, G, tb.path, epoch)
        save_optimizer(optimizer_D, D, tb.path, epoch)
        print(f"✅ Model saved at epoch {epoch}")
