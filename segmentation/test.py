import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
import albumentations as albu
import segmentation_models_pytorch as smp
import torch
from PIL import Image


def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values
    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


class Dataset(BaseDataset):

    def __init__(self, images_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)

        self.images_fps = [os.path.join(images_dir, image_id, 'img.png') for image_id in self.ids]
        self.masks_fps = [os.path.join(images_dir, image_id, 'label.png') for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        mask = np.array(Image.open(self.masks_fps[i]))

        masks = [(mask == v) for v in [0, 1, 2, 3, 4, 5]]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def make_print_to_file(path='./'):
    """
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    """
    import sys
    import os
    import config_file as cfg_file
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('segMain-Day' + '%Y_%m_%d' + '_Time' + '%H:%M:%S')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))


def get_training_augmentation():
    train_transform = [
        albu.Resize(512, 512),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


if __name__ == "__main__":
    # Visualize Sample Image and Mask
    dataset = BuildingsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)
    random_idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[2]

    visualize(
        original_image=image,
        ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask=reverse_one_hot(mask)
    )

    # Visualize Augmented Images & Masks

    augmented_dataset = BuildingsDataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )

    random_idx = random.randint(0, len(augmented_dataset) - 1)

    # Different augmentations on a random image/mask pair (256*256 crop)
    for i in range(3):
        image, mask = augmented_dataset[random_idx]
        visualize(
            original_image=image,
            ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
            one_hot_encoded_mask=reverse_one_hot(mask)
        )
    # strat------------------------------------------------------------------------------------
    make_print_to_file(path='/home/eye/XCS/aieyes/segmentation/Log')
    img_dir = "./data_img_mIoU"

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'
    # Set num of epochs
    EPOCHS = 80
    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model
    # checkpoint)
    TRAINING = True

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=6,
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        img_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = Dataset(
        img_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    if TRAINING:
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            print("Evaluation on Train Data: ")
            print(f"Mean IoU Score: {train_logs['iou_score']:.4f}")
            print(f"Mean Dice Loss: {train_logs['dice_loss']:.4f}")
            valid_logs = valid_epoch.run(valid_loader)
            print("Evaluation on Test Data: ")
            print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
            print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

            # torch.save(model, './best_model.pth')
            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, './weights/best_model_20220326.pth')
                print('Model saved!')
    # Prediction on Test Data
    # load best saved model checkpoint from the current run
    if os.path.exists('./weights/best_model.pth'):
        best_model = torch.load('./weights/best_model.pth', map_location=DEVICE)
        print('Loaded U-Net model from this run.')

    # load best saved model checkpoint from previous commit (if present)
    # elif os.path.exists('../input//deeplabv3-efficientnetb4-frontend-using-pytorch/best_model.pth'):
    #     best_model = torch.load('../input//deeplabv3-efficientnetb4-frontend-using-pytorch/best_model.pth',
    #                             map_location=DEVICE)
    #     print('Loaded DeepLabV3+ model from a previous commit.')

    # create test dataloader (with preprocessing operation: to_tensor(...))
    test_dataset = Dataset(
        img_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset)

    # test dataset for visualization (without preprocessing transformations)
    test_dataset_vis = Dataset(
        img_dir,
        augmentation=get_validation_augmentation(),
    )

    # get a random test image/mask index
    random_idx = random.randint(0, len(test_dataset_vis) - 1)
    image, mask = test_dataset_vis[random_idx]

    visualize(
        original_image=image,
        ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask=reverse_one_hot(mask)
    )

    # Center crop padded image / mask to original image dims
    def crop_image(image, target_image_dims=[1500, 1500, 3]):
        target_size = target_image_dims[0]
        image_size = len(image)
        padding = (image_size - target_size) // 2

        return image[
               padding:image_size - padding,
               padding:image_size - padding,
               :,
               ]


    sample_preds_folder = './sample_predictions/'
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    for idx in range(len(test_dataset)):
        image, gt_mask = test_dataset[idx]
        image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Get prediction channel corresponding to building
        pred_building_heatmap = pred_mask[:, :, select_classes.index('building')]
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
        # Convert gt_mask from `CHW` format to `HWC` format
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))
        cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"),
                    np.hstack([image_vis, gt_mask, pred_mask])[:, :, ::-1])

        visualize(
            original_image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pred_mask,
            predicted_building_heatmap=pred_building_heatmap
        )

    # Model Evaluation on Test Dataset

    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    valid_logs = test_epoch.run(test_dataloader)
    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

    # Plot Dice Loss & IoU Metric for Train vs. Val
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.T

    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('IoU Score', fontsize=20)
    plt.title('IoU Score Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('iou_score_plot.png')
    plt.show()

    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Dice Loss', fontsize=20)
    plt.title('Dice Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('dice_loss_plot.png')
    plt.show()
