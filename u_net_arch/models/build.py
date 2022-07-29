from multiprocessing.sharedctypes import Value
from tkinter import OFF
import torch
import torch.nn as nn


from .backbones import ResNet, ResPCPNet
from .heads import ClassifierResNet, MultiPartSegHeadResNet, SceneSegHeadResNet, MultiDimHeadResNet, DiscriminatorHead
from .losses import LabelSmoothingCrossEntropyLoss, MultiShapeCrossEntropy, MaskedCrossEntropy, MaskedL1Loss, MaskedChamferL1Loss, MaskedChamferLoss, MaskedAdaptiveL1ChamferLoss,MaskedBinaryCrossEntropy,MaskedOutlierLoss, MaskedOffsetLoss


OFFSET_REG_DIM = 3
OUTLIER_DETECT_DIM = 1

def build_classification(config):
    model = ClassificationModel(config,
                                config.backbone, config.head, config.num_classes, config.input_features_dim,
                                config.radius, config.sampleDl, config.nsamples, config.npoints,
                                config.width, config.depth, config.bottleneck_ratio)
    criterion = LabelSmoothingCrossEntropyLoss()
    return model, criterion


def build_multi_part_segmentation(config):
    model = MultiPartSegmentationModel(config, config.backbone, config.head, config.num_classes, config.num_parts,
                                       config.input_features_dim,
                                       config.radius, config.sampleDl, config.nsamples, config.npoints,
                                       config.width, config.depth, config.bottleneck_ratio)
    criterion = MultiShapeCrossEntropy(config.num_classes)
    return model, criterion


def build_scene_segmentation(config):
    model = SceneSegmentationModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)
    criterion = MaskedCrossEntropy()
    return model, criterion


def build_offset_regression(config):
    model = OffsetRegressionModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)

    print(f"Using loss {config.loss}")
    if config.loss == 'L1':
        criterion = MaskedL1Loss()
    elif config.loss == 'chamfer_L1':
        criterion = MaskedChamferL1Loss()
    elif config.loss == 'chamfer':
        criterion = MaskedChamferLoss()
    elif config.loss == 'chamfer_sparse':
        criterion = MaskedChamferLoss(norm_type='L1')
    elif config.loss == 'l1_chamfer_sparse':
        criterion = MaskedChamferL1Loss(norm_type='L1')
    elif config.loss == 'l1_chamfer_adaptive_to_chamfer':
        criterion = MaskedAdaptiveL1ChamferLoss(converging_to='chamfer')
    elif config.loss == 'l1_chamfer_adaptive_to_l1':
        criterion = MaskedAdaptiveL1ChamferLoss(converging_to='L1')
    else:
        if config.loss is None:
            raise ValueError("Please specify a loss in the config file")
        raise ValueError(f"The loss {config.loss} is not implemented")
    return model,criterion

def build_complete_denoising(config):
    model = OffsetRegOutlierDetectModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)



    if config.loss == "L1_classification":
        criterion_offset = MaskedL1Loss()
        criterion_outlier = nn.BCELoss()
    elif config.loss == "Weighted_L1_classification":
        criterion_offset = MaskedL1Loss()
        criterion_outlier = nn.BCELoss()
    elif config.loss == "double_weight":
        criterion_offset = MaskedOffsetLoss()
        criterion_outlier = MaskedOutlierLoss()
    else:
        raise ValueError(f"Loss {config.loss} not implemented.")
    
    return model,(criterion_offset,criterion_outlier)

def build_discriminator(config):
    model = DiscriminatorBlock(config, config.backbone, config.head_discriminator, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)

    criterion = nn.BCELoss()
    # criterion = LabelSmoothingCrossEntropyLoss()
    return model,criterion


def build_offset_regression_PCN(config):

    model = ResPCPNet(
        num_points=config.num_points,
        output_dim=3,
        use_feat_stn=True,
        sym_op="max")
    
    criterion = nn.L1Loss()

    return model,criterion


class ClassificationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes,
                 input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(ClassificationModel, self).__init__()

        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Classification Model")

        if head == 'resnet_cls':
            self.classifier = ClassifierResNet(num_classes, width)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Classification Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.classifier(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class MultiPartSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, num_parts,
                 input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(MultiPartSegmentationModel, self).__init__()
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_part_seg':
            self.segmentation_head = MultiPartSegHeadResNet(num_classes, width, radius, nsamples, num_parts)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class SceneSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(SceneSegmentationModel, self).__init__()
        if input_features_dim==0:
            input_features_dim=3# all ones
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_scene_seg':
            self.segmentation_head = SceneSegHeadResNet(num_classes, width, radius, nsamples)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class DiscriminatorBlock(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(DiscriminatorBlock, self).__init__()
        if input_features_dim==0:
            input_features_dim = 3# all ones
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Offset Regression Model")

        if head == 'discriminator_head':
            self.head = DiscriminatorHead(width)
        else:
            raise NotImplementedError(f"Head {head} not implemented in Offset Regression Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.head(end_points)

    def init_weights(self):
        # from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m,nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class OffsetRegressionModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(OffsetRegressionModel, self).__init__()
        if input_features_dim==0:
            input_features_dim=3# all ones
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Offset Regression Model")

        if head == 'offset_reg_head':
            self.segmentation_head = MultiDimHeadResNet(OFFSET_REG_DIM,width, radius, nsamples,isGAN=config.GAN)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Offset Regression Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

class OffsetRegOutlierDetectModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(OffsetRegOutlierDetectModel, self).__init__()
        if input_features_dim==0:
            input_features_dim=3# all ones
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Offset Regression Model")

        self.segmentation_head = MultiDimHeadResNet(OFFSET_REG_DIM+OUTLIER_DETECT_DIM,width, radius, nsamples,isGAN=config.GAN)

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)