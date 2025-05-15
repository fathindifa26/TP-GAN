# config/config_distillation.py

# === Data & training ===
train = {}
train['img_list']        = './train_list.txt'
train['learning_rate']   = 1e-4
train['num_epochs']      = 200
train['batch_size']      = 50
train['log_step']        = 1000
train['resume_model']    = None     # path ke checkpoint teacher jika ingin memuat ulang
train['resume_optimizer']= None

# === Generator (Teacher & Student share zdim, num_classes, dll) ===
G = {}
G['zdim']               = 64
G['use_residual_block'] = False
G['use_batchnorm']      = False
G['num_classes']        = 347

# === Discriminator ===
D = {}
D['use_batchnorm']      = False

# === Loss weights ===
loss = {}
loss['weight_gradient_penalty'] = 10    # untuk WGAN-GP
loss['weight_adv_G']            = 1e-3  # bobot loss adv pada student
loss['weight_distill']          = 1.0   # bobot L2 distillation antara student & teacher

# === Feature extractor (dipakai oleh distillation script) ===
feature_extract_model = {}
# path ke direktori yang berisi pretrain_config.py dan checkpoint .pth
# contoh: 'feature_extract_models/mobilenetv2'
feature_extract_model['resume'] = 'feature_extract_models/mobilenetv2'