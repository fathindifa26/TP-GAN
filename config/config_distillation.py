# config_distillation.py

# === Data & training ===
train = {}
train['img_list']        = './train_list.txt'
train['learning_rate']   = 1e-4
train['num_epochs']      = 200
train['batch_size']      = 50
train['log_step']        = 1000
train['resume_model']    = None
train['resume_optimizer']= None

# === Generator (Teacher & Student) ===
G = {}
G['zdim']               = 64
G['use_residual_block'] = False
G['use_batchnorm']      = False
G['num_classes']        = 347

# === Discriminator ===
D = {}
D['use_batchnorm'] = False

# === Loss weights ===
loss = {}
loss['weight_gradient_penalty'] = 10    # WGAN-GP
loss['weight_adv_G']            = 1e-3  # adversarial loss pada student
loss['weight_distill']          = 1.0   # bobot L2 distillation

# === Feature extractor (frozen) ===
feature_extract_model = {}
# ini harus folder yang berisi:
#   - checkpoint(.pth)
#   - pretrain_config.py
feature_extract_model['resume'] = 'save/feature_extractor_models/mobilenetv2'
