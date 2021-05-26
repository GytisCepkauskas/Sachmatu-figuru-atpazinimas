# !!! Neturėtų būti keičiami !!!
IMG_HEIGHT, IMG_WIDTH = 256, 256

# !!! Neturėtų būti keičiami, jei naudojamas „Chessman image dataset“ duomenų rinkinys !!!
TRAINING_SET_SIZE, VALIDATION_SET_SIZE = 509, 42

# „Batch“ dydis
BATCH_SIZE = 64

# Epizodų skaičius mokymui
EPOCHS = 6000

# Praretinimo tikimybė ChessNet ir OverfittingModel tinkluose
DROPOUT_CHANCE = 0.30

# Ar mokymo duomenų vaizdams yra pritaikoma augmentacija
ENABLE_IMAGE_AUGMENTATION = True

# Palaikomi modeliai „paprastam“ mokymui: ChessNet, OverfittingModel, VGG19, MobileNetV2, ResNet50V2, ResNet101V2, InceptionV3, Xception
# Palaikomi modeliai perkeltam mokymui: VGG19, MobileNetV2, ResNet50V2, ResNet101V2, InceptionV3, Xception
MODEL_TO_USE = "ChessNet"

# Fiksuotas pasukimo kampas (laipsniais), kuriu pasukami visi testiniai vaizdai
FIXED_ROTATION_DEGREE = 0

# Fiksuotas pasukimo kampas (laipsniais), kuriu pasukami visi testiniai vaizdai
FULL_ROTATION_ON = False

# Failas tinklo, kuris bus naudojamas adaptacijoje
MODEL_TO_FINE_TUNE_FILE = "Collected Models\\ChessNet.h5"

# Failas tinklo, kuris bus naudojamas vertinime arba spėjimuose
MODEL_TO_EVALUATE_FILE = "Collected Models\\ChessNet.h5"

# Pasukimo metodo alfa reikšmė (laipsniais)
ROTATION_RULE_ALPHA = 90
