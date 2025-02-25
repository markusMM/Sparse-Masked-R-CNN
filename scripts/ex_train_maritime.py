import sparse_mask_rcnn.nets as modellib
from sparse_mask_rcnn.datasets.maritime import COCO_MODEL_PATH, MODEL_DIR, MaritimeConfig, MaritimeDataset
from sparse_mask_rcnn.utils import coco


import argparse
import os


def main():
    import argparse

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        coco.download_trained_weights(COCO_MODEL_PATH)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Maritime.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset",
                        help='Directory of the Maritime dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=MODEL_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = MaritimeConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)

    if args.model.lower() == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                                     "mrcnn_bbox", "mrcnn_mask"])
    elif args.model.lower() == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    else:
        model.load_weights(args.model, by_name=True)

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = MaritimeDataset()
    dataset_train.load_maritime(args.dataset, "training")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MaritimeDataset()
    dataset_val.load_maritime(args.dataset, "validation")
    dataset_val.prepare()


    # *** This training schedule is an example. Update to your needs ***
    # Training - Stage 1
    print("Training network heads")
    model.train_model(dataset_train, dataset_val,  # type: ignore
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train_model(dataset_train, dataset_val,  # type: ignore
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train_model(dataset_train, dataset_val,  # type: ignore
                learning_rate=config.LEARNING_RATE / 10,
                epochs=20,
                layers='all')

# %% Training
if __name__ == '__main__':
    main()
