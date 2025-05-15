# config/config_distillation.py

# === Data & training ===
train = {
    'img_list':         './train_list.txt',
    'learning_rate':    1e-4,
    'num_epochs':       200,
    'batch_size':       50,
    'log_step':         1000,
    'resume_model':     None,   # path ke checkpoint teacher jika perlu load ulang
    'resume_optimizer': None
}

# === Generator (Teacher & Student) ===
G = {
    'zdim':              64,
    'use_residual_block': False,
    'use_batchnorm':      False,
    'num_classes':        347
}

# === Discriminator ===
D = {
    'use_batchnorm': False
}

# === Loss weights ===
loss = {
    'weight_gradient_penalty': 10,    # WGAN-GP
    'weight_adv_G':            1e-3,  # adversarial loss pada student
    'weight_distill':          1.0    # bobot L2 distillation
}

# === Feature extractor ===
feature_extract_model = {
    # Path ke folder berisi .pth checkpoint (mis: 'feature_extract_models/mobilenetv2')
    'resume':     'feature_extract_models/mobilenetv2',
    # Nama fungsi di models/feature_extract_network.py
    'model_name': 'mobilenetv2',
    # Argumen untuk inisialisasi extractor
    'kwargs': {
        'num_classes': 347,
        'input_size':  128,
        'width_mult':  1.0,
        'dropout':     0.5,
        'pretrained':  False
    }
}
