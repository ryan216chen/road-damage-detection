from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2 
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

NUM_CLASSES = 5 

def build_model():

    weights = (
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT 
    )

    model = (
        fasterrcnn_resnet50_fpn_v2(
            weights = weights 
        )
    )

    in_features = (
        model
        .roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )

    return model 