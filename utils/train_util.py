from segmentation_models_pytorch.utils.train import Epoch
import albumentations as albu


class TrainEpoch(Epoch):
    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        scheduler=None,
        device="cpu",
        verbose=True,
    ):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return loss, prediction


def get_training_augmentation():
    train_transform = [
        albu.Resize(height=434, width=636),
        albu.PadIfNeeded(min_height=448, min_width=640),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.1, rotate_limit=5, shift_limit=0.05, p=1, border_mode=0
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    valid_transform = [
        albu.Resize(height=434, width=636),
        albu.PadIfNeeded(min_height=448, min_width=640),
    ]
    return albu.Compose(valid_transform)


def get_tta_augmentation():
    tta_transform = [
        albu.Resize(height=434, width=636),
        albu.PadIfNeeded(min_height=448, min_width=640),
        albu.HorizontalFlip(always_apply=True),
    ]
    return albu.Compose(tta_transform)


def to_tensor_img(x, **kwargs):
    x = x.transpose(2, 0, 1).astype("float32") / 255.0
    return x


def to_tensor_mask(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing():
    _transform = [albu.Lambda(image=to_tensor_img, mask=to_tensor_mask)]
    return albu.Compose(_transform)
