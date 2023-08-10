import torch
import torch.nn as nn
import models.group1 as group1_models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if "resnet" in model_name:
        """ Resnet34, 50, 101, 152'
        """
        model_ft = group1_models.__dict__[model_name](pretrained=use_pretrained).cuda()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "mobilenet" in model_name or 'mnasnet' in model_name:
        """ mobilenet_v2, MNasNet
        """
        model_ft = group1_models.__dict__[model_name](pretrained=use_pretrained).cuda()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "densenet" in model_name:
        """ DenseNet
        """
        model_ft = group1_models.__dict__[model_name](pretrained=use_pretrained).cuda()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "inception" in model_name:
        """ InceptionV3
        """
        aux_logits = False
        model_ft = group1_models.__dict__[model_name](pretrained=True, aux_logits=aux_logits).cuda()
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        if aux_logits:
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif "googlenet" in model_name:
        """ InceptionV3
        """
        aux_logits = False
        model_ft = group1_models.__dict__[model_name](pretrained=True, aux_logits=aux_logits).cuda()
        set_parameter_requires_grad(model_ft, feature_extract)
        if aux_logits:
            # Handle the auxilary net
            num_ftrs = model_ft.aux1.fc2.in_features
            model_ft.aux1.fc2 = nn.Linear(num_ftrs, num_classes)

            num_ftrs = model_ft.aux2.fc2.in_features
            model_ft.aux2.fc2 = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


if __name__ == '__main__':
    # Initialize the model for this run
    model_name = 'densenet'
    num_classes = 100
    feature_extract = False
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)
