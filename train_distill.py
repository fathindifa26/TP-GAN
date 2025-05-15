import torch
from torchvision import transforms
import config.config_distillation as config
from data.data import TrainDataset
import numpy as np
from models.network import Discriminator, Generator, GeneratorStudent
from torch.autograd import Variable
import time
from logs.log import TensorBoardX
from utils.utils import *
from models import feature_extract_network
import importlib

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.Tensor.pin_memory = lambda self: self

test_time = False

if __name__ == "__main__":
    # --- DataLoader ---
    img_list = open(config.train['img_list'], 'r').read().split('\n')
    img_list.pop()
    dataloader = torch.utils.data.DataLoader(
        TrainDataset(img_list),
        batch_size=config.train['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # --- Teacher (frozen) ---
    teacher = torch.nn.DataParallel(
        Generator(
            zdim=config.G['zdim'],
            use_batchnorm=config.G['use_batchnorm'],
            use_residual_block=config.G['use_residual_block'],
            num_classes=config.G['num_classes']
        )
    ).cuda()
    if config.train['resume_model']:
        resume_model(teacher, config.train['resume_model'], strict=False)
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Student (light) ---
    student = torch.nn.DataParallel(
        GeneratorStudent(
            zdim=config.G['zdim'],
            num_classes=config.G['num_classes'],
            use_batchnorm=config.G['use_batchnorm'],
            use_residual_block=config.G['use_residual_block'],
            fm_mult=0.75
        )
    ).cuda()

    # --- Discriminator ---
    D = torch.nn.DataParallel(
        Discriminator(use_batchnorm=config.D['use_batchnorm'])
    ).cuda()

    # --- Optimizers ---
    optimizer_S = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=config.train['learning_rate']
    )
    optimizer_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, D.parameters()),
        lr=config.train['learning_rate']
    )

    # --- TensorBoard ---
    tb = TensorBoardX(config_filename_list=["config_distillation.py"])

    # --- Feature extractor (frozen) ---
    folder = config.feature_extract_model['resume']
    pretrain_cfg = importlib.import_module('.'.join([*folder.split('/'), 'pretrain_config']))
    model_name = pretrain_cfg.stem['model_name']
    kwargs     = pretrain_cfg.stem.copy()
    kwargs.pop('model_name')
    feat = eval('feature_extract_network.' + model_name)(**kwargs)
    resume_model(feat, folder, strict=False)
    feat = torch.nn.DataParallel(feat).cuda()
    for p in feat.parameters():
        p.requires_grad = False

    # --- Loss ---
    mse = torch.nn.MSELoss().cuda()

    # --- Training loop ---
    for epoch in range(config.train['num_epochs']):
        for step, batch in enumerate(dataloader):
            # pindahkan semua tensor ke GPU
            for k in batch:
                batch[k] = Variable(batch[k].cuda(non_blocking=True), requires_grad=False)
            z = Variable(torch.randn(len(batch['img']), config.G['zdim'])).cuda()

            # ---- Teacher forward ----
            with torch.no_grad():
                t128, t64, t32, *_ = teacher(
                    batch['img'], batch['img64'], batch['img32'],
                    batch['left_eye'], batch['right_eye'],
                    batch['nose'], batch['mouth'],
                    z, use_dropout=False
                )

            # ---- Student forward & distill loss ----
            s128, s64, s32, *_ = student(
                batch['img'], batch['img64'], batch['img32'],
                batch['left_eye'], batch['right_eye'],
                batch['nose'], batch['mouth'],
                z, use_dropout=True
            )
            distill_loss = mse(s128, t128) + mse(s64, t64) + mse(s32, t32)

            # ---- Update D ----
            set_requires_grad(D, True)
            adv_D_loss = -torch.mean(D(batch['img_frontal'])) + torch.mean(D(s128.detach()))
            alpha = torch.rand(batch['img_frontal'].size(0), 1, 1, 1).cuda()
            interp = Variable(alpha * s128.detach() + (1 - alpha) * batch['img_frontal'],
                              requires_grad=True)
            out = D(interp)
            grad = torch.autograd.grad(
                outputs=out, inputs=interp,
                grad_outputs=torch.ones_like(out),
                retain_graph=True, create_graph=True
            )[0].view(out.size(0), -1)
            gp = torch.mean((grad.norm(2, 1) - 1) ** 2)
            L_D = adv_D_loss + config.loss['weight_gradient_penalty'] * gp
            optimizer_D.zero_grad()
            L_D.backward()
            optimizer_D.step()

            # ---- Update student ----
            set_requires_grad(D, False)
            adv_S_loss = -torch.mean(D(s128))
            L_S = config.loss['weight_distill'] * distill_loss + config.loss['weight_adv_G'] * adv_S_loss
            optimizer_S.zero_grad()
            L_S.backward()
            optimizer_S.step()

            # ---- Logging ----
            if step % config.train['log_step'] == 0:
                global_step = epoch * len(dataloader) + step
                print(f"[Epoch {epoch} | Step {step}/{len(dataloader)}] "
                      f"Distill: {distill_loss.item():.4f}, Adv_S: {adv_S_loss.item():.4f}")
                tb.add_scalar("distill_loss", distill_loss.item(), global_step, 'train')
                tb.add_scalar("adv_student", adv_S_loss.item(), global_step, 'train')
                tb.add_image_grid("grid/teacher", 4, t128 * 0.5 + 0.5, global_step, 'train')
                tb.add_image_grid("grid/student", 4, s128 * 0.5 + 0.5, global_step, 'train')

        # simpan student setiap epoch
        save_model(student, tb.path, epoch)
        print(f"✅ Saved student @ epoch {epoch}")
