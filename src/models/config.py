class confiFile():
    """Configuration class for easy parametrization"""
    
    #Pretrained model with timm
    model = 'resnet10t'
    epochs = 2
    
    in_chans = 3
    num_classes = 10
    learning_rate = 1e-3
    
    val_size = 0.3
    batch_size = 8